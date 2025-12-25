from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import tempfile
import shutil
import httpx
import asyncio

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="Alzheimer's Voice Biomarker Analysis API")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Import classifier (lazy loading)
classifier = None

def get_classifier():
    global classifier
    if classifier is None:
        from svm_classifier import AlzheimersSVMClassifier
        classifier = AlzheimersSVMClassifier(model_path=str(ROOT_DIR / 'models'))
    return classifier


# Define Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StatusCheckCreate(BaseModel):
    client_name: str


class PredictionResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    prediction: int
    label: str
    confidence: float
    probability_healthy: float
    probability_alzheimers: float
    risk_level: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TrainingRequest(BaseModel):
    audio_urls: List[str]
    labels: List[int]  # 0 = Healthy, 1 = Alzheimer's
    optimize: bool = True
    n_features: int = 50


class TrainingResponse(BaseModel):
    status: str
    message: str
    n_samples: int
    n_features_selected: int
    selected_features: List[str]


class ModelInfo(BaseModel):
    status: str
    model_type: Optional[str] = None
    kernel: Optional[str] = None
    n_support_vectors: Optional[int] = None
    n_features_selected: Optional[int] = None
    selected_features: Optional[List[str]] = None
    class_labels: Optional[List[str]] = None


# API Routes
@api_router.get("/")
async def root():
    return {
        "message": "Alzheimer's Voice Biomarker Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/api/predict - Analyze audio file for Alzheimer's biomarkers",
            "train": "/api/train - Train the SVM model with labeled data",
            "model_info": "/api/model/info - Get trained model information",
            "predictions": "/api/predictions - Get prediction history"
        }
    }


@api_router.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    """
    Analyze an audio file for Alzheimer's voice biomarkers.
    Accepts .wav files.
    """
    if not file.filename.endswith(('.wav', '.WAV')):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")
    
    clf = get_classifier()
    
    # Check if model is trained
    if not clf.is_trained and not clf.load_model():
        raise HTTPException(
            status_code=400, 
            detail="Model not trained. Please train the model first using /api/train endpoint."
        )
    
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Make prediction
        result = clf.predict(temp_path)
        
        # Store prediction in database
        prediction_doc = {
            'id': str(uuid.uuid4()),
            'filename': file.filename,
            'prediction': result['prediction'],
            'label': result['label'],
            'confidence': result['confidence'],
            'probability_healthy': result['probability_healthy'],
            'probability_alzheimers': result['probability_alzheimers'],
            'risk_level': result['risk_level'],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        await db.predictions.insert_one(prediction_doc)
        
        return {
            'status': 'success',
            'filename': file.filename,
            **result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


@api_router.post("/predict/url")
async def predict_from_url(url: str = Form(...)):
    """
    Analyze an audio file from URL for Alzheimer's voice biomarkers.
    """
    clf = get_classifier()
    
    if not clf.is_trained and not clf.load_model():
        raise HTTPException(
            status_code=400, 
            detail="Model not trained. Please train the model first."
        )
    
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "audio.wav")
    
    try:
        # Download audio file
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=60.0)
            response.raise_for_status()
            
            with open(temp_path, "wb") as f:
                f.write(response.content)
        
        # Make prediction
        result = clf.predict(temp_path)
        
        # Store prediction
        prediction_doc = {
            'id': str(uuid.uuid4()),
            'filename': url.split('/')[-1],
            'source_url': url,
            'prediction': result['prediction'],
            'label': result['label'],
            'confidence': result['confidence'],
            'probability_healthy': result['probability_healthy'],
            'probability_alzheimers': result['probability_alzheimers'],
            'risk_level': result['risk_level'],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        await db.predictions.insert_one(prediction_doc)
        
        return {
            'status': 'success',
            'source': url,
            **result
        }
        
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Error downloading audio: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@api_router.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train the SVM classifier with labeled audio data.
    
    - audio_urls: List of URLs to .wav audio files
    - labels: List of labels (0 = Healthy, 1 = Alzheimer's)
    - optimize: Whether to perform hyperparameter optimization (default: True)
    - n_features: Number of features to select (default: 50)
    """
    if len(request.audio_urls) != len(request.labels):
        raise HTTPException(
            status_code=400, 
            detail="Number of audio URLs must match number of labels"
        )
    
    if len(request.audio_urls) < 2:
        raise HTTPException(
            status_code=400, 
            detail="Need at least 2 samples for training"
        )
    
    clf = get_classifier()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Download all audio files
        audio_files = []
        async with httpx.AsyncClient() as http_client:
            for i, url in enumerate(request.audio_urls):
                try:
                    response = await http_client.get(url, timeout=60.0)
                    response.raise_for_status()
                    
                    filename = f"audio_{i}.wav"
                    filepath = os.path.join(temp_dir, filename)
                    
                    with open(filepath, "wb") as f:
                        f.write(response.content)
                    
                    audio_files.append(filepath)
                except Exception as e:
                    logging.error(f"Failed to download {url}: {str(e)}")
                    continue
        
        if len(audio_files) < 2:
            raise HTTPException(
                status_code=400, 
                detail="Could not download enough valid audio files"
            )
        
        # Train the model
        result = clf.train(
            audio_files=audio_files,
            labels=request.labels[:len(audio_files)],
            optimize=request.optimize,
            n_features=request.n_features
        )
        
        # Store training info in database
        training_doc = {
            'id': str(uuid.uuid4()),
            'n_samples': result['n_samples'],
            'n_features_selected': result['n_features_selected'],
            'selected_features': result['selected_features'],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        await db.training_history.insert_one(training_doc)
        
        return TrainingResponse(
            status='success',
            message=f"Model trained successfully with {result['n_samples']} samples",
            n_samples=result['n_samples'],
            n_features_selected=result['n_features_selected'],
            selected_features=result['selected_features']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@api_router.post("/train/files")
async def train_from_files(
    files: List[UploadFile] = File(...),
    labels: str = Form(...)  # Comma-separated labels
):
    """
    Train the SVM classifier with uploaded audio files.
    
    - files: Multiple .wav audio files
    - labels: Comma-separated labels (0 = Healthy, 1 = Alzheimer's)
    """
    label_list = [int(l.strip()) for l in labels.split(',')]
    
    if len(files) != len(label_list):
        raise HTTPException(
            status_code=400, 
            detail="Number of files must match number of labels"
        )
    
    clf = get_classifier()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded files
        audio_files = []
        for i, file in enumerate(files):
            if not file.filename.endswith(('.wav', '.WAV')):
                continue
            
            filepath = os.path.join(temp_dir, f"audio_{i}.wav")
            with open(filepath, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            audio_files.append(filepath)
        
        if len(audio_files) < 2:
            raise HTTPException(
                status_code=400, 
                detail="Need at least 2 valid .wav files"
            )
        
        # Train
        result = clf.train(
            audio_files=audio_files,
            labels=label_list[:len(audio_files)],
            optimize=True,
            n_features=50
        )
        
        return {
            'status': 'success',
            'message': f"Model trained with {result['n_samples']} samples",
            **result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@api_router.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the trained model"""
    clf = get_classifier()
    
    if not clf.is_trained:
        clf.load_model()
    
    return clf.get_model_info()


@api_router.get("/predictions", response_model=List[PredictionResult])
async def get_predictions(limit: int = 100):
    """Get prediction history"""
    predictions = await db.predictions.find(
        {}, 
        {"_id": 0}
    ).sort("timestamp", -1).limit(limit).to_list(limit)
    
    return predictions


@api_router.delete("/predictions")
async def clear_predictions():
    """Clear prediction history"""
    result = await db.predictions.delete_many({})
    return {"deleted": result.deleted_count}


@api_router.get("/features/info")
async def get_features_info():
    """Get information about the 101 voice biomarker features"""
    return {
        "total_features": 101,
        "categories": {
            "spectral_features": {
                "count": 52,
                "description": "Frequency-domain features from audio spectrum",
                "features": [
                    "MFCCs (39): 13 coefficients × 3 (mean, delta, delta-delta)",
                    "Spectral Centroid: Center of mass of spectrum",
                    "Spectral Bandwidth: Width of spectral band",
                    "Spectral Contrast (7): Difference between peaks and valleys",
                    "Spectral Flatness: Measure of tone vs noise",
                    "Spectral Rolloff: Frequency below which 85% of energy lies",
                    "Spectral Flux: Rate of spectral change",
                    "Zero Crossing Rate: Rate of signal sign changes"
                ]
            },
            "temporal_features": {
                "count": 25,
                "description": "Time-domain features analyzing speech patterns",
                "features": [
                    "Pause patterns (6): Count, mean/std/max/min/total duration",
                    "Speech segments (3): Count, mean/std duration",
                    "Speech-to-pause ratio",
                    "RMS energy (5): Mean, std, max, min, range",
                    "Energy entropy",
                    "Temporal envelope (4): Mean, std, skewness, kurtosis",
                    "Estimated syllable rate",
                    "Temporal flatness"
                ]
            },
            "pitch_prosody": {
                "count": 10,
                "description": "Pitch and intonation characteristics",
                "features": [
                    "Pitch mean, std, max, min, range",
                    "Pitch monotonicity",
                    "Pitch slope (mean, std)",
                    "Voiced ratio",
                    "Pitch coefficient of variation"
                ]
            },
            "voice_quality": {
                "count": 10,
                "description": "Voice quality measurements",
                "features": [
                    "Jitter (3): Local, RAP, PPQ5",
                    "Shimmer (3): Local, APQ3, APQ5",
                    "HNR (2): Mean, std (Harmonics-to-Noise Ratio)",
                    "Breathiness"
                ]
            },
            "speech_timing": {
                "count": 4,
                "description": "Overall speech timing measurements",
                "features": [
                    "Total duration",
                    "Phonation time",
                    "Phonation time ratio",
                    "Articulation rate"
                ]
            }
        },
        "biomarker_significance": {
            "alzheimers_indicators": [
                "Increased pause frequency and duration",
                "Reduced speech rate",
                "Higher pitch variability or monotonicity",
                "Increased jitter and shimmer",
                "Lower HNR (more noise in voice)",
                "Changes in spectral characteristics"
            ]
        }
    }


@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    _ = await db.status_checks.insert_one(doc)
    return status_obj


@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    return status_checks


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
