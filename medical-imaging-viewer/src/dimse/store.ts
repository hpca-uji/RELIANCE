import { ConfParams, config } from '../utils/config';
import { startStoreScp, storeScpOptions, shutdownScu, shutdownScuOptions } from 'dicom-dimse-native';
import { FastifyBaseLogger } from 'fastify/types/logger';

export async function startScp(log: FastifyBaseLogger) {
  const ts = config.get(ConfParams.XTRANSFER) as string;

  const options: storeScpOptions = {
    source: config.get(ConfParams.SOURCE),
    peers: config.get(ConfParams.PEERS),
    storagePath: config.get(ConfParams.STORAGE_PATH),
    netTransferPrefer: ts,
    netTransferPropose: ts,
    writeTransfer: ts,
    permissive: true,
    verbose: config.get(ConfParams.VERBOSE),
  };
  log.info(`pacs-server listening on port: ${options.source?.port}`);

  startStoreScp(options, (result: string) => {
    // currently this will never log
    //log.info(JSON.parse(result));
    log.info(`store-scp started in source ${JSON.stringify(options.source, null, 2)}`);
  });
}

export async function shutdown(log: FastifyBaseLogger) {
  const options: shutdownScuOptions = {
    source: config.get(ConfParams.SOURCE),
    target: config.get(ConfParams.SOURCE),
    verbose: config.get(ConfParams.VERBOSE),
  };

  log.info(`sending shutdown request to target: ${options.target?.aet}`);

  return new Promise((resolve, reject) => {
    shutdownScu(options, (result: string) => {
      if (result && result.length > 0) {
        try {
          //log.info(JSON.parse(result));
          log.info(`${options.target?.aet} shutdown completed`);
          resolve(true);
        } catch (error) {
          log.error(result);
          reject(error);
        }
      }
      log.info(`no scu to shutdown`);
      reject();
    });
  });
}
