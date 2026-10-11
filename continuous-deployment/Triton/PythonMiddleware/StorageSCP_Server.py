from pynetdicom import AE, evt, AllStoragePresentationContexts, debug_logger
import numpy as np
import DicomPreprocessor as DmP
import grpcTritonClient
import httpTritonClient
import json
import DB.dbImplementation as dbI
import pathlib
from datetime import datetime

URL_TRITON='http://localhost:8000/v2/models'

#debug_logger()

# Implement a handler for evt.EVT_C_STORE
def handle_store(event):
    """Handle a C-STORE request event."""
    try:
        print("DICOM received")
        # Decode the C-STORE request's *Data Set* parameter to a pydicom Dataset
        ds = event.dataset
        # Add the File Meta Information
        ds.file_meta = event.file_meta
        # Get the image from DICOM file
        data = DmP.loadDataFromDicom(ds)
        # Calculate receivedtime
        receivedtime = datetime.now()
        # Make the prediction
        grpcPrediction = grpcTritonClient.inferInTritonTLS(data,
                                model_name="model",
                                url="localhost:8001",
                                verbose=False,
                                label_file=pathlib.Path(pathlib.Path(__file__).parent.resolve().as_posix(),"labels/chest/labels.txt"))
        # Calculate processedtime
        processedtime = datetime.now()
        # Get acquisitiontime from DICOM
        #acquisitiontime = None
        acquisitiontime = datetime.fromtimestamp(int(ds.get(0x00080022).value) + int(ds.get(0x00080032).value))
        # Insert the prediction in the DB
        dbI.insertInference(ds.SOPInstanceUID, json.dumps(grpcPrediction), receivedtime, processedtime, acquisitiontime=acquisitiontime)
        print(json.dumps(grpcPrediction, indent=4, sort_keys=True))
        # Save the dataset using the SOP Instance UID as the filename
        #ds.save_as(ds.SOPInstanceUID, write_like_original=False)
    except Exception as e:
        print(e)
        
    # Return a 'Success' status
    return 0x0000

handlers = [(evt.EVT_C_STORE, handle_store)]

# Initialise the Application Entity
ae = AE(ae_title="INFERENCE_SCP")

# Support presentation contexts for all storage SOP Classes
ae.supported_contexts = AllStoragePresentationContexts
# Start listening for incoming association requests
#ae.start_server(("0.0.0.0", 11112), evt_handlers=handlers)
ae.start_server(("0.0.0.0", 104), evt_handlers=handlers)
