"""
Voice Feature Extraction Module for Alzheimer's Detection
Extracts 101 biomarker features from speech audio files
"""

import numpy as np
import librosa
import scipy.signal as signal
from scipy.stats import kurtosis, skew
import warnings
warnings.filterwarnings('ignore')


class VoiceFeatureExtractor:
    """
    Extracts 101 voice biomarker features for Alzheimer's detection:
    - Spectral Features (52): MFCCs, spectral moments, etc.
    - Temporal Features (25): Pause patterns, speech rate, etc.
    - Pitch/Prosody (10): Pitch variation, monotonicity, etc.
    - Voice Quality (10): Jitter, shimmer, HNR, etc.
    - Speech Timing (4): Duration, articulation rate, etc.
    """
    
    def __init__(self, sr=22050, n_mfcc=13, n_mels=128):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        
    def extract_all_features(self, audio_path):
        """Extract all 101 features from an audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # Normalize audio
            y = librosa.util.normalize(y)
            
            features = {}
            
            # 1. Spectral Features (52 features)
            spectral_features = self._extract_spectral_features(y, sr)
            features.update(spectral_features)
            
            # 2. Temporal Features (25 features)
            temporal_features = self._extract_temporal_features(y, sr)
            features.update(temporal_features)
            
            # 3. Pitch/Prosody Features (10 features)
            pitch_features = self._extract_pitch_features(y, sr)
            features.update(pitch_features)
            
            # 4. Voice Quality Features (10 features)
            voice_quality_features = self._extract_voice_quality_features(y, sr)
            features.update(voice_quality_features)
            
            # 5. Speech Timing Features (4 features)
            timing_features = self._extract_timing_features(y, sr)
            features.update(timing_features)
            
            return features
            
        except Exception as e:
            raise ValueError(f"Error extracting features: {str(e)}")
    
    def _extract_spectral_features(self, y, sr):
        """Extract 52 spectral features"""
        features = {}
        
        # MFCCs (39 features: 13 coefficients x 3 - mean, delta, delta-delta)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        for i in range(self.n_mfcc):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_delta_mean'] = np.mean(mfcc_delta[i])
            features[f'mfcc_{i+1}_delta2_mean'] = np.mean(mfcc_delta2[i])
        
        # Spectral Centroid (1 feature)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        
        # Spectral Bandwidth (1 feature)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        
        # Spectral Contrast (7 features - 6 bands + 1 mean)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast_{i+1}'] = np.mean(spectral_contrast[i])
        
        # Spectral Flatness (1 feature)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        features['spectral_flatness_mean'] = np.mean(spectral_flatness)
        
        # Spectral Rolloff (1 feature)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        
        # Spectral Flux (1 feature)
        spectral_flux = np.sqrt(np.sum(np.diff(np.abs(librosa.stft(y)))**2, axis=0))
        features['spectral_flux_mean'] = np.mean(spectral_flux)
        
        # Zero Crossing Rate (1 feature)
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate_mean'] = np.mean(zcr)
        
        return features
    
    def _extract_temporal_features(self, y, sr):
        """Extract 25 temporal features"""
        features = {}
        
        # Detect speech and silence segments
        intervals = librosa.effects.split(y, top_db=25)
        
        # Pause features
        if len(intervals) > 1:
            pause_durations = []
            for i in range(len(intervals) - 1):
                pause_start = intervals[i][1]
                pause_end = intervals[i + 1][0]
                pause_duration = (pause_end - pause_start) / sr
                pause_durations.append(pause_duration)
            
            pause_durations = np.array(pause_durations)
            features['pause_count'] = len(pause_durations)
            features['pause_mean_duration'] = np.mean(pause_durations) if len(pause_durations) > 0 else 0
            features['pause_std_duration'] = np.std(pause_durations) if len(pause_durations) > 0 else 0
            features['pause_max_duration'] = np.max(pause_durations) if len(pause_durations) > 0 else 0
            features['pause_min_duration'] = np.min(pause_durations) if len(pause_durations) > 0 else 0
            features['pause_total_duration'] = np.sum(pause_durations) if len(pause_durations) > 0 else 0
        else:
            features['pause_count'] = 0
            features['pause_mean_duration'] = 0
            features['pause_std_duration'] = 0
            features['pause_max_duration'] = 0
            features['pause_min_duration'] = 0
            features['pause_total_duration'] = 0
        
        # Speech segment features
        speech_durations = [(interval[1] - interval[0]) / sr for interval in intervals]
        features['speech_segment_count'] = len(speech_durations)
        features['speech_segment_mean_duration'] = np.mean(speech_durations) if len(speech_durations) > 0 else 0
        features['speech_segment_std_duration'] = np.std(speech_durations) if len(speech_durations) > 0 else 0
        
        # Speech-to-pause ratio
        total_speech = sum(speech_durations)
        total_pause = features.get('pause_total_duration', 0)
        features['speech_to_pause_ratio'] = total_speech / (total_pause + 1e-6)
        
        # RMS energy features (for detecting hesitations)
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_max'] = np.max(rms)
        features['rms_min'] = np.min(rms)
        features['rms_range'] = features['rms_max'] - features['rms_min']
        
        # Energy entropy (measure of speech regularity)
        rms_normalized = rms / (np.sum(rms) + 1e-6)
        features['energy_entropy'] = -np.sum(rms_normalized * np.log2(rms_normalized + 1e-6))
        
        # Temporal envelope features
        envelope = np.abs(librosa.stft(y)).mean(axis=0)
        features['envelope_mean'] = np.mean(envelope)
        features['envelope_std'] = np.std(envelope)
        features['envelope_skewness'] = skew(envelope)
        features['envelope_kurtosis'] = kurtosis(envelope)
        
        # Speech rate approximation (based on amplitude peaks)
        peaks, _ = signal.find_peaks(rms, distance=int(sr * 0.1 / 512))  # ~100ms between peaks
        duration_seconds = len(y) / sr
        features['estimated_syllable_rate'] = len(peaks) / duration_seconds
        
        # Additional temporal statistics
        features['temporal_flatness'] = np.min(rms) / (np.max(rms) + 1e-6)
        
        return features
    
    def _extract_pitch_features(self, y, sr):
        """Extract 10 pitch/prosody features"""
        features = {}
        
        # Extract pitch using librosa's pyin
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Remove unvoiced frames (NaN values)
        f0_voiced = f0[~np.isnan(f0)]
        
        if len(f0_voiced) > 0:
            features['pitch_mean'] = np.mean(f0_voiced)
            features['pitch_std'] = np.std(f0_voiced)
            features['pitch_max'] = np.max(f0_voiced)
            features['pitch_min'] = np.min(f0_voiced)
            features['pitch_range'] = features['pitch_max'] - features['pitch_min']
            
            # Monotonicity (low variation = monotonic)
            features['pitch_monotonicity'] = 1.0 / (features['pitch_std'] + 1e-6)
            
            # Pitch contour features
            if len(f0_voiced) > 1:
                pitch_diff = np.diff(f0_voiced)
                features['pitch_slope_mean'] = np.mean(pitch_diff)
                features['pitch_slope_std'] = np.std(pitch_diff)
            else:
                features['pitch_slope_mean'] = 0
                features['pitch_slope_std'] = 0
            
            # Voiced ratio
            features['voiced_ratio'] = len(f0_voiced) / len(f0)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_max'] = 0
            features['pitch_min'] = 0
            features['pitch_range'] = 0
            features['pitch_monotonicity'] = 0
            features['pitch_slope_mean'] = 0
            features['pitch_slope_std'] = 0
            features['voiced_ratio'] = 0
        
        # Pitch coefficient of variation
        features['pitch_cv'] = features['pitch_std'] / (features['pitch_mean'] + 1e-6)
        
        return features
    
    def _extract_voice_quality_features(self, y, sr):
        """Extract 10 voice quality features (jitter, shimmer, HNR, etc.)"""
        features = {}
        
        # Extract pitch for jitter/shimmer calculation
        f0, voiced_flag, _ = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        f0_voiced = f0[~np.isnan(f0)]
        
        if len(f0_voiced) > 1:
            # Jitter (frequency perturbation) - local jitter
            periods = 1.0 / (f0_voiced + 1e-6)
            jitter_abs = np.mean(np.abs(np.diff(periods)))
            features['jitter_local'] = jitter_abs / (np.mean(periods) + 1e-6)
            
            # Jitter RAP (Relative Average Perturbation)
            if len(periods) > 2:
                rap = np.abs(periods[1:-1] - (periods[:-2] + periods[1:-1] + periods[2:]) / 3)
                features['jitter_rap'] = np.mean(rap) / (np.mean(periods) + 1e-6)
            else:
                features['jitter_rap'] = 0
            
            # Jitter PPQ5 (Five-point Period Perturbation Quotient)
            if len(periods) > 4:
                ppq5 = []
                for i in range(2, len(periods) - 2):
                    local_avg = np.mean(periods[i-2:i+3])
                    ppq5.append(np.abs(periods[i] - local_avg))
                features['jitter_ppq5'] = np.mean(ppq5) / (np.mean(periods) + 1e-6)
            else:
                features['jitter_ppq5'] = 0
        else:
            features['jitter_local'] = 0
            features['jitter_rap'] = 0
            features['jitter_ppq5'] = 0
        
        # Shimmer (amplitude perturbation)
        rms_frames = librosa.feature.rms(y=y)[0]
        if len(rms_frames) > 1:
            shimmer_abs = np.mean(np.abs(np.diff(rms_frames)))
            features['shimmer_local'] = shimmer_abs / (np.mean(rms_frames) + 1e-6)
            
            # Shimmer APQ3
            if len(rms_frames) > 2:
                apq3 = []
                for i in range(1, len(rms_frames) - 1):
                    local_avg = np.mean(rms_frames[i-1:i+2])
                    apq3.append(np.abs(rms_frames[i] - local_avg))
                features['shimmer_apq3'] = np.mean(apq3) / (np.mean(rms_frames) + 1e-6)
            else:
                features['shimmer_apq3'] = 0
            
            # Shimmer APQ5
            if len(rms_frames) > 4:
                apq5 = []
                for i in range(2, len(rms_frames) - 2):
                    local_avg = np.mean(rms_frames[i-2:i+3])
                    apq5.append(np.abs(rms_frames[i] - local_avg))
                features['shimmer_apq5'] = np.mean(apq5) / (np.mean(rms_frames) + 1e-6)
            else:
                features['shimmer_apq5'] = 0
        else:
            features['shimmer_local'] = 0
            features['shimmer_apq3'] = 0
            features['shimmer_apq5'] = 0
        
        # HNR (Harmonics-to-Noise Ratio) approximation
        # Using autocorrelation method
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)  # 10ms hop
        
        hnr_values = []
        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i:i + frame_length]
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            if len(autocorr) > 1 and autocorr[0] > 0:
                # Find first peak after zero crossing
                peak_idx = np.argmax(autocorr[1:]) + 1
                if autocorr[peak_idx] > 0:
                    hnr = 10 * np.log10(autocorr[peak_idx] / (autocorr[0] - autocorr[peak_idx] + 1e-6) + 1e-6)
                    hnr_values.append(hnr)
        
        features['hnr_mean'] = np.mean(hnr_values) if len(hnr_values) > 0 else 0
        features['hnr_std'] = np.std(hnr_values) if len(hnr_values) > 0 else 0
        
        # Breathiness (inverse of HNR, high breathiness = low HNR)
        features['breathiness'] = 1.0 / (features['hnr_mean'] + 1e-6) if features['hnr_mean'] > 0 else 0
        
        return features
    
    def _extract_timing_features(self, y, sr):
        """Extract 4 speech timing features"""
        features = {}
        
        # Total speech duration
        features['total_duration'] = len(y) / sr
        
        # Detect speech segments
        intervals = librosa.effects.split(y, top_db=25)
        
        # Phonation time (total voiced time)
        phonation_time = sum([(interval[1] - interval[0]) / sr for interval in intervals])
        features['phonation_time'] = phonation_time
        
        # Phonation time ratio
        features['phonation_time_ratio'] = phonation_time / features['total_duration']
        
        # Articulation rate (speech segments per second of speech)
        if phonation_time > 0:
            features['articulation_rate'] = len(intervals) / phonation_time
        else:
            features['articulation_rate'] = 0
        
        return features
    
    def get_feature_vector(self, audio_path):
        """Get features as a numpy array for ML model input"""
        features = self.extract_all_features(audio_path)
        return np.array(list(features.values())), list(features.keys())


# Helper function for feature extraction
def extract_features_from_file(audio_path):
    """Convenience function to extract features from a single file"""
    extractor = VoiceFeatureExtractor()
    return extractor.extract_all_features(audio_path)
