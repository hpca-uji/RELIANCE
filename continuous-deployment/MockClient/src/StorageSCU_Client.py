from pydicom import dcmread
from pynetdicom import AE, debug_logger, StoragePresentationContexts
from pynetdicom.sop_class import CTImageStorage
import pathlib
import os

#debug_logger()

# Initialise the Application Entity
ae = AE(ae_title="DICOM_MOCK_SCU")

# Add a requested presentation context
#ae.add_requested_context(CTImageStorage)
ae.requested_contexts = StoragePresentationContexts

# Read in our DICOM CT dataset
dicomPath = pathlib.Path(os.environ['MOCK_DICOM_FILE'])
print(dicomPath)
ds = dcmread(dicomPath)

# Associate with the configured local DICOM receiver.
assoc = ae.associate(os.environ.get("DICOM_DEST_HOST", "localhost"),
                     int(os.environ.get("DICOM_DEST_PORT", "104")))
if assoc.is_established:
    # Use the C-STORE service to send the dataset
    # returns the response status as a pydicom Dataset
    status = assoc.send_c_store(ds)

    # Check the status of the storage request
    if status:
        # If the storage request succeeded this will be 0x0000
        print('C-STORE request status: 0x{0:04x}'.format(status.Status))
    else:
        print('Connection timed out, was aborted or received invalid response')

    # Release the association
    assoc.release()
else:
    print('Association rejected, aborted or never connected')
