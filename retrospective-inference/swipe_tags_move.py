import atexit
import abc
import argparse
import json
import pathlib
import logging.config
import logging.handlers
import os
import numpy as np
import pandas as pd
import asyncio
import aiomysql
import pydicom
import pynetdicom
import torch
from typing import Dict, List, Tuple, NoReturn
from threading import Lock
from ssl import SSLContext
from dataclasses import dataclass

# # # # # # # # # # # # # # #
# BEiT model                #
# # # # # # # # # # # # # # #
class InferenceModel(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, '_preprocess_data') and
                    callable(subclass._preprocess_data) and
                    hasattr(subclass, '_post_process_prediction') and
                    callable(subclass._post_process_prediction) and
                    hasattr(subclass, 'run_inference') and
                    callable(subclass.run_inference) or
                    NotImplemented)
    @abc.abstractmethod
    async def _preprocess_data(self, ds: pydicom.Dataset) -> Tuple[np.ndarray, bool, bool]:
        """ Preprocess input data to match whatever format the inference model expects. """
        raise NotImplementedError
    @abc.abstractmethod
    async def _post_process_prediction(self, results: torch.Tensor, assumed_color_scheme: bool, assumed_bits_stored: bool) -> Dict[str, str]:
        """ Post-process prediction results. """
        raise NotImplementedError
    @abc.abstractmethod
    async def run_inference(self, ds: pydicom.Dataset) -> Dict[str, str]:
        """ Use the preloaded model to perform an inference based on the input DICOM Dataset. """
        raise NotImplementedError
class ChestBEiTModel:
    def __init__(self, model_path: pathlib.Path):
        self._model_path = model_path
        self._model = torch.jit.load(self._model_path)
    
    async def _preprocess_data(self, ds: pydicom.Dataset) -> Tuple[np.ndarray, bool, bool]:
        """
        Extract image from DICOM file and apply the required transformations.

        Args:
            ds (pydicom.Dataset): The DICOM file to extract the data from.

        Returns:
            numpy.ndarray: DICOM image data preprocessed and ready to be forwarded to the model.
            bool: The color scheme is not present in DICOM file and has been assumed.
            bool: The bits stored value is not present in DICOM file and has been assumed.
        """
        # Read array and tags
        color_scheme = ds.get(0x20500020)
        bits_stored = ds.get(0x00280101)
        
        # Read pixel_data
        img = ds.pixel_array
        
        assumed_color_scheme = False
        # Check for Nones
        if color_scheme is None:
            color_scheme = "IDENTITY" # Assume Identity by default
            assumed_color_scheme = True
        else:
            color_scheme = color_scheme.value
        
        assumed_bits_stored = False
        if bits_stored is None:
            bits_stored = 16 # Assume 16 by default
            assumed_bits_stored = True
        else:
            bits_stored = bits_stored.value
        
        # Convert to range 0-255 if needed
        if bits_stored != 8:
            img = (img // (2**(bits_stored-8)))
            img = img.astype(np.uint8)
        img = torch.ByteTensor(img)

        # Inverse color if needed
        if color_scheme == "INVERSE":
            img = 255-img
        
        # Expand channel dimension and batch dimension
        img = img.unsqueeze(0).unsqueeze(0)

        # Convert to [0-1] range and return
        img = img/255.0

        return img, assumed_color_scheme, assumed_bits_stored
    async def _post_process_prediction(self, results: torch.Tensor, assumed_color_scheme: bool, assumed_bits_stored: bool) -> Dict[str, str]:
        """
        Match prediction values to corresponding labels.
        It is assumed that results are ordered to match labels in labels file.

        Args:
            results (torch.Tensor): The prediction results from the model.
            assumed_color_scheme (bool): The color scheme is not present in DICOM file and has been assumed.
            assumed_bits_stored (bool): The bits stored value is not present in DICOM file and has been assumed.
        
        Returns:
            Dict[str, str]: Dictionary containing key/value pairs with the inference results.
            Key is the labels name and value is the prediction percentage.
        """
        results: List[float] = results.tolist()
        labels_predictions: Dict[str, str] = {}
        for (idx, prediction) in enumerate(results):
            labels_predictions[f"out{idx}"] = str(prediction)
        
        labels_predictions['Assumed Color Scheme'] = assumed_color_scheme
        labels_predictions['Assumed Bits Stored'] = assumed_bits_stored

        return labels_predictions 
    async def run_inference(self, ds: pydicom.Dataset) -> Dict[str, str]:
        """
        Use the preloaded model to perform an inference based on the input DICOM Dataset.

        Args:
            ds (pydicom.Dataset): DICOM Dataset to perform inference on.

        Returns:
            Dict[str, str]: Dictionary containing key/value pairs with the inference results.
            Key is the labels name and value is the prediction percentage.
        """
        # Preprocess DICOM image
        (preprocessed_data, assumed_color_scheme, assumed_bits_stored) = await self._preprocess_data(ds)
        # Run inference
        with torch.no_grad():
            results: torch.Tensor = self._model(preprocessed_data)
        # Post process prediction results
        return await self._post_process_prediction(results[0], assumed_color_scheme, assumed_bits_stored)

# # # # # # # # # # # # # # #
# DB connection             #
# # # # # # # # # # # # # # #
@dataclass(frozen=True)
class StudyIdentifier:
    dicom_id: int
    accession_number: str
    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str
class DBPool:
    def __init__(self, logger: logging.Logger):
        self._cnxpool = None
        self._logger = logger
        self._connection_config_dict: Dict[str, str | int] = {
            'user': os.environ['DB_USER'],
            'password': os.environ['DB_PWD'],
            'host': os.environ.get('DB_HOST', 'localhost'),
            'port': int(os.environ.get('DB_PORT', '3306')),
            'db': os.environ.get('DB_NAME', 'app_db')
        }
    
    async def create_connection_pool(self, loop: asyncio.AbstractEventLoop) -> aiomysql.Pool | None:
        """
        Create a connection pool attached to an asynchronous loop.

        Args:
            loop (asyncio.AbstractEventLoop): Event loop to attach the pool to.

        Returns:
            aiomysql.Pool: The connection pool, where individual connections can be retrieved in order to operate with the DB.
            None: Nothing, if the connection could not be established with DB.
        """
        if self._cnxpool:
            return self._cnxpool
        
        try:
            self._cnxpool: aiomysql.Pool = await aiomysql.create_pool(minsize=2, maxsize=10, loop=loop, autocommit=False, **self._connection_config_dict)
        except Exception as e:
            self._logger.error(f'Error while connecting to MySQL {e}')

        return self._cnxpool
    async def close_connection_pool(self) -> None: # NoReturn
        """
        Close connection pool, if it is open.
        """
        if not self._cnxpool:
            return
        self._cnxpool.close()
        await self._cnxpool.wait_closed()
class DBImplementation:
    def __init__(self, logger: logging.Logger):
        self._logger = logger
    
    async def read_studies(self, pool: aiomysql.Pool, reviewer_user_id: int, limit: int = 5, offset: int = 0) -> List[StudyIdentifier]:
        """
        Insert a prediction into the DB.

        Args:
            pool (aiomysql.Pool): Pool to get connection from.
        
        Keyword args:
            limit (int): Limit how many rows are retrieved from the Database (default 5).
            offset (int): Offset of the rows to retrieve from the Database (default 0).
        """
        result = None

        async with pool.acquire() as connection:
            async with connection.cursor() as cursor:
                # Start transaction
                await connection.begin()
                # Query the DB
                query = (
                    "SELECT d.DicomID, d.AccessionNumber, d.StudyInstanceUID, d.SeriesInstanceUID, d.SOPInstanceUID "
                    "FROM Dicoms AS d INNER JOIN RadiologistPredictions AS rp ON d.DicomID = rp.DicomID "
                    "WHERE rp.UserID = %s AND d.IsDiscarded = 0 "
                    "ORDER BY d.DicomID ASC "
                    "LIMIT %s "
                    "OFFSET %s "
                )
                await cursor.execute(query, (reviewer_user_id, limit, offset))
                result = await cursor.fetchall()
                result = [StudyIdentifier(*si) for si in result]
                # Finish transaction
                await connection.commit()
                self._logger.debug(f'Successfully read {limit} study identifiers starting from {offset}')
        
        return result

# # # # # # # # # # # # # # #
# Store results handler     #
# # # # # # # # # # # # # # #
class ResultsHandler(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'store_results') and 
                callable(subclass.store_results) or 
                NotImplemented)

    @abc.abstractmethod
    def store_results(self, results: Dict[str, str]) -> NoReturn:
        """ Store results. """
        raise NotImplementedError
class DumpToCSVHandler(ResultsHandler):
    def __init__(self, logger: logging.Logger, output_name: str = "results", fresh_start: bool = True):
        self._logger = logger
        self._output_name = output_name
        self._fresh_start = fresh_start
        self._filename = pathlib.Path(pathlib.Path(__file__).parent.resolve().as_posix(),f"results/{self._output_name}.csv").as_posix()
        self._lock = Lock()
    
    def _store_to_csv(self, dataframe: pd.DataFrame, filename: str, mode: str = 'w', header: bool = False) -> None: # NoReturn
        """
        Store input dataframe into a CSV file.

        Args:
            dataframe (pandas.DataFrame): Dataframe to be stored.
            filename (str): Filename (and path) of the output file.
        
        Keyword args:
            mode (str): Overwrite ('w') or append ('a') data to file, if it exists (Default 'w')
            header (bool): Maintain dataframes header when writing to file (Default False)
        """
        dataframe.to_csv(filename, sep='\t', encoding='utf-8', mode=mode, header=header, index=False)
    def store_results(self, results: Dict[str, str]) -> NoReturn:
        """
        Dump results into a CSV file. Reinitialize and append header to CSV file the first time.
        Append to existing CSV file without header in subsequent calls.
        """
        with self._lock:
            if self._fresh_start:
                self._logger.debug("Initializing CSV...")
                self._store_to_csv(pd.DataFrame([results]), self._filename, mode='w', header=True)
                self._fresh_start = False
            else:
                self._store_to_csv(pd.DataFrame([results]), self._filename, mode='a')

# # # # # # # # # # # # # # #
# DICOM SCP Server          #
# # # # # # # # # # # # # # #
@dataclass(frozen=True)
class ModalitySCP:
    address: Tuple[str, int]
    ae_title: str
    ssl_context: SSLContext
class ServerInit:
    def __init__(self, modality, context: List[pynetdicom.presentation.PresentationContext], handler: List[pynetdicom.events.EventHandlerType]):
        self._modality = modality
        self._context = context
        self._handler = handler

    async def __aenter__(self):
        ae = pynetdicom.AE()
        ae.supported_contexts = self._context
        self._server = ae.start_server(**vars(self._modality), evt_handlers=self._handler, block=False)
        return self._server
    async def __aexit__(self, *args):
        if self._server:
            self._server.shutdown()
        self._server = None
    def __enter__(self):
        ae = pynetdicom.AE()
        ae.supported_contexts = self._context
        self._server = ae.start_server(**vars(self._modality), evt_handlers=self._handler, block=False)
        return self._server
    def __exit__(self, *args):
        if self._server:
            self._server.shutdown()
        self._server = None
class SCPServer:
    def __init__(
            self,
            inference_models: List[InferenceModel],
            results_handlers: List[ResultsHandler],
            loop,
            study_identifiers: pd.DataFrame,
            logger: logging.Logger,
            num_workers: int = 10,
            work_queue_max_size: int = 100
    ):
        self._inference_models = inference_models
        self._results_handlers = results_handlers
        # Initialize loggers
        self._logger = logger
        # Initialize DICOM configuration
        self._init_modality()
        self._init_context()
        self._init_handlers()
        # Define server loop
        self._loop = loop
        # Save study identifiers
        self._study_identifiers = study_identifiers
        # create the work queue
        self._num_workers = num_workers
        self._work_queue = asyncio.Queue(maxsize=work_queue_max_size)
    # Modality
    def _init_modality(self) -> None: # NoReturn
        ssl_context = None
        self._modality = ModalitySCP(('0.0.0.0', 11113), 'INFERENCE_SCP', ssl_context)
    # Context
    def _initialize_scp_contexts(
        self,
        raw_contexts: List[pynetdicom.presentation.PresentationContext] = pynetdicom.StoragePresentationContexts,
        transfer_syntaxes: List[str] = pynetdicom.DEFAULT_TRANSFER_SYNTAXES + [pydicom.uid.JPEG2000Lossless],
        add_verification: bool = True
    ) -> List[pynetdicom.presentation.PresentationContext]:
        contexts = []

        for context in raw_contexts:
            contexts.append(pynetdicom.build_context(context.abstract_syntax, transfer_syntaxes))

        if add_verification:
            contexts.append(pynetdicom.build_context(pynetdicom.sop_class.Verification))

        return contexts
    def _init_context(self) -> None: # NoReturn:
        self._contexts = self._initialize_scp_contexts()
    # Handlers
    def _handle_open(self, event: pynetdicom.events.Event):
        """
        Handler that prints sender's IP address when an association is established.
        Debugging purposes only.
        """
        msg = 'Connected with remote at {}'.format(event.address)
        self._logger.debug(msg)
    def _handle_echo(self, event: pynetdicom.events.Event):
        """
        Handle C-ECHO operations (DICOM ping).
        """
        self._logger.debug(f"Received C-ECHO with event context: {event.context}")
        return 0x0000
    def _extract_dicom(self, event: pynetdicom.events.Event) -> pydicom.Dataset:
        """
        Extract and decompress, if necessary, the DICOM file.

        Args:
            event (pynetdicom.events.Event): Incoming association event, containing the information (DICOM dataset, e.g.).

        Returns:
            pydicom.Dataset: Decompressed DICOM file.
        """
        # Decode the C-STORE request's *Data Set* parameter to a pydicom Dataset
        ds = event.dataset
        # Add the File Meta Information
        ds.file_meta = event.file_meta
        # Decompress file
        if ds.file_meta.TransferSyntaxUID.is_compressed is True:
            self._logger.debug("decompress")
            ds.decompress('gdcm')

        return ds
    def _create_dicom_path(self, ds: pydicom.Dataset) -> pathlib.Path:
        return pathlib.Path(pathlib.Path(__file__).parent.resolve().as_posix(), f"dicoms/{ds.SOPInstanceUID}")
    def _handle_store(
            self, event: pynetdicom.events.Event
    ):
        """
        Wait for incoming C-MOVE association and act as Producer.
        Extracts the DICOM file from the event and stores it in disk.
        The path to the file is pushed into the working queue so they are later processed by the consumers.

        Args:
            event (pynetdicom.events.Event): Incoming association event, containing the information (DICOM dataset, e.g.).
        """
        self._logger.debug(f"Received C-STORE {event.context.abstract_syntax} and {event.context.transfer_syntax}")
        try:
            ds = self._extract_dicom(event)
            # Build path
            dicom_path = self._create_dicom_path(ds)
            # Save the dataset using the SOP Instance UID as the filename
            ds.save_as(dicom_path, write_like_original=False)
            # Add dataset to the queue
            self._loop.call_soon_threadsafe(self._work_queue.put_nowait, (dicom_path,))
        except RuntimeError:
            self._logger.error('Loop has already been closed!')
        # Release connection
        return 0x0000
    def _init_handlers(self) -> None: # NoReturn:
        self._handlers = [
            (pynetdicom.evt.EVT_CONN_OPEN, self._handle_open),
            (pynetdicom.evt.EVT_C_STORE, self._handle_store),
            (pynetdicom.evt.EVT_C_ECHO, self._handle_echo)
        ]
    # Consumer
    def _delete_dicom(self, dicom_path: pathlib.Path):
        if dicom_path.exists():
            dicom_path.unlink()
    def _build_results(self, identifiers: Dict[str, str], predictions: Dict[str, str]) -> Dict[str, str]:
        """
        Merge identifiers with predictions, so the predictions can be identified.

        Args:
            identifiers (Dict[str, str]): Results identifiers. Example: {'dicom_id': '1'}
            predictions (Dict[str, str]): Predictions as key/value pairs. Example: {'Lung': '0.9832442', 'Urgent lung finding': '0.7832344'}

        Returns:
            Dict[str, str]: Predictions with a DICOM identifier. Example: {'dicom_id': '1', 'Lung': '0.9832442', 'Urgent lung finding': '0.7832344'}
        """
        return {**identifiers, **predictions}
    async def _get_local_identifiers_from_dicom(self, ds: pydicom.Dataset, lock: Lock) -> Dict[str, str]:
        """
        Get the local DB identifier of the DICOM.
        Remove the row with the local DB identifier from self._study_identifiers

        Args:
            ds (pydicom.Dataset): DICOM file to be identified.
            lock (threading.Lock): Lock so the consumers can safely access and modify the shared resources.

        Return:
            Dict[str, str]: Identifiers that, on their own, do not allow reidentificating the DICOM.
        """
        accession_number = getattr(ds, 'AccessionNumber', None)
        study_instance_uid = getattr(ds, 'StudyInstanceUID', None)
        series_instance_uid = getattr(ds, 'SeriesInstanceUID', None)
        sop_instance_uid = getattr(ds, 'SOPInstanceUID', None)

        # Find DicomID from DB results, based on DICOM identifiers
        result: pd.Series = self._study_identifiers.loc[
            self._study_identifiers['sop_instance_uid'] == sop_instance_uid,
            'dicom_id'
        ]
        if result.empty:
            self._logger.warning(f"Missing DICOM with identifiers {accession_number=} {study_instance_uid=} {series_instance_uid=} {sop_instance_uid=}")
            return {'dicom_id': f"Created{accession_number}"}
        
        dicom_id = result.values[0]

        # Delete row from dataset
        with lock:
            self._study_identifiers = self._study_identifiers[self._study_identifiers.dicom_id != dicom_id]

        return {'dicom_id': dicom_id}
    def _read_dicom(self, dicom_path: pathlib.Path):
        return pydicom.dcmread(dicom_path)
    async def _process_dicom(self, dicom_path: pathlib.Path, inference_models: List[InferenceModel], results_handlers: List[ResultsHandler], lock: Lock) -> bool:
        """
        Handle a C-STORE request event.
        
        Args:
            dicom_path (pathlib.Path): Path to the DICOM file.
            inference_model: The model that will be used for prediction
            results_handler (ResultsHandler): Class that determines how to store the results (db, csv...)
            lock (threading.Lock): Lock so the consumers can safely access and modify the shared resources.
        """
        try:
            ds = self._read_dicom(dicom_path)
            # Define coroutines
            coros = [
                self._get_local_identifiers_from_dicom(ds, lock) # Get identifiers from Dicom
                , *[model.run_inference(ds) for model in inference_models] # Make prediction with all inference models
            ]
            # Execute coroutines
            group = asyncio.gather(*coros)
            # Gather results from coroutines
            identifiers, *predictions = await group
            
            # Append results to csv/database
            for prediction, result_handler in zip(predictions, results_handlers):
                # Merge identifiers and prediction + label
                results = self._build_results(identifiers, prediction)
                result_handler.store_results(results)
            # Delete DICOM file from disk
            self._delete_dicom(dicom_path)
            return True
        except Exception as e:
            self._logger.critical(e)
        
        return False
    async def _check_finish(self, lock: Lock) -> bool:
        """
        Check if all studies have already been received.

        Args:
            lock (threading.Lock): Lock so the consumers can safely access and modify the shared resources.
        """
        lock.acquire()
        finished = self._study_identifiers.empty
        lock.release()
        if finished:
            self._logger.info("Study Identifiers dataset is empty. Work is finished, stopping server...")
        
        return finished
    async def _process_one(self, models: List[InferenceModel], results_handlers: List[ResultsHandler], lock: Lock) -> None: # NoReturn
        """
        Wait while the queue is empty. When a job is inserted in the queue, take and process it.
        
        Args:
            model: The model that will be used for prediction
            results_handler (ResultsHandler): Class that determines how to store the results (db, csv...)
            lock (threading.Lock): Lock so the consumers can safely access and modify the shared resources.
        """
        # wait to grab an item from the queue
        (dicom_path,) = await self._work_queue.get()
        try:
            # dataset in queue, lets process it
            await self._process_dicom(dicom_path, models, results_handlers, lock)
        except asyncio.CancelledError:
            return
        finally:
            # notify the queue that the task has been successfully processed
            self._logger.debug("Job done")
            self._work_queue.task_done()
    async def _consumer(self, models: List[InferenceModel], results_handlers: List[ResultsHandler], lock: Lock, timeout: int = 600) -> NoReturn:
        """
        Consumer implementation.

        Args:
            model: The model that will be used for prediction
            results_handler (ResultsHandler): Class that determines how to store the results (db, csv...)
            lock (threading.Lock): Lock so the consumers can safely access and modify the shared resources.
        """
        self._logger.info("Consumer initialized")
        finished = False
        while not finished:
            try:
                async with asyncio.timeout(timeout):
                    await self._process_one(models, results_handlers, lock)
                finished = await self._check_finish(lock)
            #except asyncio.TimeoutError: # <- Prior to python v3.11
            except TimeoutError:
                self._logger.info(f"No association established within {timeout // 60}m{timeout % 60}s. Stopping server...")
                finished = True
            except asyncio.CancelledError:
                self._logger.info("Consumer cancelled")
                return
    # Stop DICOM server
    async def _stop(self) -> None: # NoReturn
        """
        Wait until work queue is empty.
        """
        self._logger.info("Closing server...")
        await self._work_queue.join()
        self._logger.info("Bye!")
    # Initialize DICOM server
    async def listen(self) -> NoReturn:
        """
        Initialize a producer/consumer based DICOM SCP Server.
        The server will be stopped when one of the consumers confirms that all studies have
        been received or if explicitly stated (e.g. Ctrl+C Keyboard Interrupt event).
        """
        lock = Lock()
        # Create consumers
        workers = [
            asyncio.create_task(self._consumer(self._inference_models, self._results_handlers, lock))
            for _ in range(self._num_workers)
        ]
        # Start SCP Server
        async with ServerInit(self._modality, self._contexts, self._handlers):
            # Start consumers
            await asyncio.wait(workers, return_when=asyncio.FIRST_COMPLETED) # Execution stopped here
            # await asyncio.gather(*workers) # Execution stopped here
        
        # Server finished or has been stopped, stop and clean all
        await self._stop()
        # Cancel worker execution, if still pending
        for worker in workers:
            worker.cancel()
        # Wait for all workers to be finished
        await asyncio.gather(*workers)

# # # # # # # # # # # # # # #
# DICOM SCU Client          #
# # # # # # # # # # # # # # #
@dataclass(frozen=True)
class ModalitySCU:
    addr: str
    port: int
    ae_title: str
class Association:
    def __init__(self, modality, context):
        self._modality = modality
        self._context = context

    async def __aenter__(self):
        ae = pynetdicom.AE('INFERENCE_CLIENT')
        # Add a requested presentation context
        ae.add_requested_context(self._context)
        self._association = ae.associate(**vars(self._modality))
        return self._association
    async def __aexit__(self, *args):
        self._association.release()
        self._association = None
    def __enter__(self):
        ae = pynetdicom.AE('INFERENCE_CLIENT')
        # Add a requested presentation context
        ae.add_requested_context(self._context)
        self._association = ae.associate(**vars(self._modality))
        return self._association
    def __exit__(self, *args):
        self._association.release()
        self._association = None
class SCUClient:
    def __init__(
            self,
            logger: logging.Logger,
            sleep_seconds: int = 10,
            ae_title: str = 'INFERENCE_CLIENT',
            *,
            ae_called: str,
            address: str,
            port: int = 104
    ):
        # Initialize loggers
        self._logger = logger

        # Initialize private configuration variables
        self._sleep_seconds = sleep_seconds
        self._ae_title = ae_title
        self._ae_called = ae_called
        self._address = address
        self._port = port

        # Initialize DICOM configuration
        self._init_modality()
        self._init_context()
    # Modality
    def _init_modality(self) -> None: # NoReturn
        self._modality = ModalitySCU(self._address, self._port, self._ae_called)
    # Context
    def _init_context(self) -> None: # NoReturn
        self._contexts = pynetdicom.sop_class.StudyRootQueryRetrieveInformationModelMove
    
    def _split_work(self, input_list: List[StudyIdentifier], n: int) -> List[List[StudyIdentifier]]:
        """
        Split a list into roughly equal-sized parts. If length of the input list is not divisible
        by "n", last group will be smaller, containining the remainding elements up to "n".

        Args:
            input_list (List[StudyIdentifier]): The input list to be split.
            n (int): The number of elements each group has to be.

        Returns:
            List[List[StudyIdentifier]]: A list of groups (lists) containing the split parts of the input list.
        """
        #return [input_list[i:i+n] for i in range(0, len(input_list), n)] # Slower
        return np.array_split(input_list, np.ceil(len(input_list)/n))

    # Ask for DICOM datasets to PACS
    def _create_dataset(self, study: StudyIdentifier, query_retrieve_level: str = 'STUDY') -> pydicom.Dataset:
        """
        Create a DICOM dataset using the keys that identify it.

        Args:
            study (StudyIdentifier): The keys of the study that identify it.
            query_retrieve_level (str): Level of DICOM C-MOVE protocol. Possible values are ['PATIENT', 'STUDY', 'SERIES', 'IMAGE']. (Default 'STUDY')
        
        Returns:
            pydicom.Dataset: A DICOM dataset containing the keys.
        """
        identifier = pydicom.Dataset()
        identifier.AccessionNumber = study.accession_number
        identifier.StudyInstanceUID = study.study_instance_uid
        identifier.SeriesInstanceUID = study.series_instance_uid
        identifier.SOPInstanceUID = study.sop_instance_uid
        identifier.QueryRetrieveLevel = query_retrieve_level

        return identifier
    async def request_images(self, study_identifiers: List[StudyIdentifier], groups: int = 4) -> NoReturn:
        """
        Request images to the PACS server.
        
        Problem: Since the implementation of pynetdicom's C-MOVE protocol is synchronous, the whole execution
        will be stopped while asking for DICOM files. Therefore, the SCP server will not be receiving the
        DICOM files until the SCU is finished.
        
        Solution: The DICOM studies to ask for will be split into groups. For each group, the SCU will
        open a new association and ask for all the DICOMs in the current group. Afterwards, it will
        sleep for a specified amount of time so the execution thread is finally assigned to the SCP
        and it processes the incoming DICOM files.

        Args:
            study_identifiers (List[StudyIdentifier]): A list containing all the study identifiers to request to the PACS.
        
        Keyword args:
            groups (int): Amount of study identifiers each group should have (Default 4)
        """
        # Split work to be able to launch it as async
        work_groups = self._split_work(study_identifiers, groups)
        for work in work_groups:
            # Connect to PACS
            async with Association(self._modality, self._contexts) as assoc:
                if not assoc.is_established:
                    self._logger.critical("Association rejected, aborted or never connected")
                    self._logger.critical("Could not establish a connection with modality")
                    return

                # Ask for all DICOMs in current group
                for s_ident in work:
                    # Create dataset to send to PACS with identifiers
                    dataset = self._create_dataset(s_ident, query_retrieve_level='IMAGE')
                    # Ask PACS for the DICOM. It is BLOCKING, not async
                    responses = assoc.send_c_move(dataset, self._ae_title, self._contexts)
                    for (status, identifier) in responses:
                        if status:
                            self._logger.debug('C-MOVE query status: 0x{0:04x}'.format(status.Status))
            # Sleep SCU, so the SCP is able to catch up
            await asyncio.sleep(self._sleep_seconds)

# # # # # # # # # # # # # # #
# Load inference models     #
# # # # # # # # # # # # # # #
def get_files_with_extension(folder_path: pathlib.Path, extension: str = ".pt") -> List[pathlib.Path]:
    return [file for file in folder_path.iterdir() if file.suffix == extension]

# # # # # # # # # # # # # # #
# Loggers                   #
# # # # # # # # # # # # # # #
def setup_logging(logging_name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(logging_name)
    config_file = pathlib.Path(pathlib.Path(__file__).parent.resolve().as_posix(), "logging_configs/config.json")
    with open(config_file) as f_in:
        config = json.load(f_in)

    logging.config.dictConfig(config)
    queue_handler = logging.getHandlerByName("queue_handler")
    if queue_handler is not None:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)
    
    return logger

# # # # # # # # # # # # # # #
# Argument parser           #
# # # # # # # # # # # # # # #
def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION]...",
        description="Run inference of DICOM files stored in PACS.",
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 1.0.0"
    )
    
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers that will run inference (default: 1)')
    parser.add_argument('--work-queue-max-size', type=int, default=100, help='Maximum size of the SCP server queue (default: 100)')
    parser.add_argument('--study-limit', type=int, default=12, help='Number of studies to make inference from (default: 12)')
    parser.add_argument('--study-offset', type=int, default=0, help='Offset of studies to retrieve from DB (default: 0)')
    parser.add_argument('--fresh-start', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Mantain results from previous executions')
    parser.add_argument('--models-repository', type=str, default="models-repository", help='Folder where models are stored. It is relative to the current files folder (default: models-repository)')
    parser.add_argument('--models-extension', type=str, default=".pt", help='Models extension (default: .pt)')
    parser.add_argument('--scu-wait-time', type=int, default=10, help='Seconds the SCU will wait per model so the SCP is able to catch up (default: 10)')
    parser.add_argument('--scu-group-size', type=int, default=4, help='Size of the group the SCU will ask for before sleeping for the --scu-wait-time seconds (default: 4)')

    return parser

# # # # # # # # # # # # # # #
# Main                      #
# # # # # # # # # # # # # # #
async def main() -> NoReturn:
    # Capture arguments
    parser = init_argparse()
    args = parser.parse_args()
    # Set up logging
    logger = setup_logging()
    # Create DB pool
    dbpool = DBPool(logger)
    pool = await dbpool.create_connection_pool(asyncio.get_event_loop())
    if not pool:
        # Could not establish connection with the DB, finish execution
        return
    # Query DB and get all study identifiers
    study_identifiers: List[StudyIdentifier] = await DBImplementation(logger).read_studies(
        pool,
        reviewer_user_id=int(os.environ['REVIEWER_USER_ID']),
        limit=args.study_limit,
        offset=args.study_offset
    )

    # Close DB pool
    await dbpool.close_connection_pool()

    # Read inference models
    folder_path = pathlib.Path(pathlib.Path(__file__).parent.resolve().as_posix(), args.models_repository)
    models_path = get_files_with_extension(folder_path, args.models_extension)
    inference_models = [ChestBEiTModel(path) for path in models_path]
    results_handlers = [DumpToCSVHandler(logger, path.stem, args.fresh_start) for path in models_path]
    # DICOM SCP Server
    scp = SCPServer(
        inference_models,
        results_handlers,
        asyncio.get_event_loop(),
        pd.DataFrame([vars(s_id) for s_id in study_identifiers]), # convert List[StudyIdentifiers] to pandas DataFrame
        logger,
        args.num_workers,
        args.work_queue_max_size
    )
    # DICOM SCU Client
    sleep_seconds = len(inference_models) * args.scu_wait_time # seconds for each inference model
    scu = SCUClient(
        logger,
        sleep_seconds=sleep_seconds,
        ae_called=os.environ['DICOM_PEER_AE'],
        address=os.environ['DICOM_PEER_HOST'],
        port=int(os.environ.get('DICOM_PEER_PORT', '104'))
    )

    # Create Server and Client coroutines
    tasks = [
        asyncio.create_task(scp.listen()),
        asyncio.create_task(scu.request_images(study_identifiers, groups=args.scu_group_size))
    ]
    # Start Server and Client
    await asyncio.gather(*tasks)
if __name__ == '__main__':
    try:
        print("Press Ctrl+C to stop server...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Keyboard Interrupt captured. Bye!")
