import { ConfParams, config } from '../utils/config';
import { DicomsRepository } from '../db/models'
import { IDicoms } from '../db/tables';
import { IDataBaseConfiguration, IConfigurationPeers } from '../utils/configurationInterfaces';
import moment from 'moment';
import { FastifyInstance } from 'fastify';

interface IQueryParams {
  [key: string]: string;
}
interface IDicomIdentificators {
  StudyInstanceUID: string;
  SeriesInstanceUID: string;
  SopInstanceUID: string;
}
interface IOHIFFilters {
  limit: number | undefined;
  offset: number | undefined;
  accessionNumber: string | undefined;
  studyInstanceUID: string | undefined;
  seriesInstanceUID: string | undefined;
  sopInstanceUID: string | undefined;
  receivedTime: {startDate: string, endDate: string} | undefined;
  predictionTime: {startDate: string, endDate: string} | undefined;
  timesLabeled: number | undefined;
}

function mysqlDateToOHIF(mysqlDate: Date): string {
  if (mysqlDate == null){
    return '';
  }
  const monthNames = [
    'ene',
    'feb',
    'mar',
    'abr',
    'may',
    'jun',
    'jul',
    'ago',
    'sept',
    'oct',
    'nov',
    'dic',
  ];
  const mm = mysqlDate.getMonth(); // getMonth() is zero-based
  const dd = mysqlDate.getDate();

  return (
    monthNames[mm] +
    '. ' +
    (dd > 9 ? '' : '0') + dd +
    ', ' +
    mysqlDate.getFullYear()
  );
}

function parseStudyDate(studyDate: string | undefined): {startDate: string, endDate: string} | undefined{
  if (studyDate === undefined) return undefined;

  const parsedDates = studyDate.split('-');
  const startDate: string = moment(parsedDates[0]).format('YYYY-MM-DD');
  const endDate: string = moment(parsedDates[1]).add(1, 'days').format('YYYY-MM-DD');

  return {
    startDate: startDate,
    endDate: endDate
  };
}

function extractFiltersFromQuery(query: IQueryParams): IOHIFFilters {
  return {
    limit: query.limit ? parseInt(query.limit, 10) : 25,
    offset: query.offset ? parseInt(query.offset, 10) : 0,
    accessionNumber: query.AccessionNumber ? (config.get(ConfParams.APPEND_WILDCARD) ? query.AccessionNumber += '*': query.AccessionNumber): undefined,
    studyInstanceUID: query?.StudyInstanceUID,
    seriesInstanceUID: query?.SeriesInstanceUID,
    sopInstanceUID: query?.SopInstanceUID,
    receivedTime: parseStudyDate(query?.StudyDate),
    predictionTime: parseStudyDate(query?.DiagnoseDate),
    timesLabeled: query.TimesLabeled ? parseInt(query.TimesLabeled, 10) : undefined,
  }
}

interface IDicomMetadata {
  "00080050": {
    "Value": [string | undefined];
    "vr": string;
  };
  "0020000D": {
    "Value": [string | undefined];
    "vr": string;
  };
  "00080018": {
    "Value": [string | undefined];
    "vr": string;
  };
  "0020000E": {
    "Value": [string | undefined];
    "vr": string;
  };
  DiagnoseDate: [string];
  ReceivedDate: [string];
  TimesLabeled: [number];
}

export async function doFind(server: FastifyInstance, query: IQueryParams, userId: string): Promise<IDicomMetadata[][]> {
  const peers = config.get(ConfParams.PEERS) as IConfigurationPeers[];
  const promises: Array<Promise<IDicomMetadata[]>> = [];

  peers.forEach((peer) => {
    promises.push(sendCFindRequest(server, peer.db, query, userId));
  });
  return Promise.all(promises);
}

export async function sendCFindRequest(server: FastifyInstance, target: IDataBaseConfiguration, query: IQueryParams, userId: string): Promise<IDicomMetadata[]> {
  const ohifFilters = extractFiltersFromQuery(query);
  
  // run sql query and return json response
  return DicomsRepository.readAllPendingOrdered(server, userId, ohifFilters)
      .then((result: IDicoms[]) => {
      // cumplimiento
      const resultJsons = Promise.all(result.map(async (iDicom): Promise<IDicomMetadata> => {
        const DiagnoseDate = mysqlDateToOHIF(iDicom.DiagnoseDate);
        const ReceivedDate = mysqlDateToOHIF(iDicom.InsertedAt);
        return <IDicomMetadata> {
          // AccessionNumber
          "00080050": {
            "Value": [iDicom.AccessionNumber],
            "vr": "SH"
          },
          // StudyInstanceUID
          "0020000D": {
            "Value": [iDicom.StudyInstanceUID],
            "vr": "UI"
          },
          // SopInstanceUID
          "00080018": {
            "Value": [iDicom.SOPInstanceUID],
            "vr": "UI"
          },
          // SeriesInstanceUID
          "0020000E": {
            "Value": [iDicom.SeriesInstanceUID],
            "vr": "UI"
          },
          // Custom tags
          'DiagnoseDate': [DiagnoseDate],
          'ReceivedDate': [ReceivedDate],
          'TimesLabeled': [iDicom.TimesLabeled],
        };
      }));
      return resultJsons;
    }, (err: string) => {
      // rechazo
      server.log.error(err);
      server.log.error('invalid result received');
      return [];
    });
}
