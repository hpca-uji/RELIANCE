import { ConfParams, config } from '../utils/config';
import { echoScu, echoScuOptions, Node as DicomNode } from 'dicom-dimse-native';
import { FastifyBaseLogger } from 'fastify/types/logger';

export async function sendEcho(log: FastifyBaseLogger) {
  const peers = config.get(ConfParams.PEERS) as DicomNode[];

  const promises: Array<Promise<unknown>> = [];

  peers.forEach((peer) => {
    promises.push(sendCEchoRequest(peer, log));
  });

  return Promise.all(promises);
}

export async function sendCEchoRequest(target: DicomNode, log: FastifyBaseLogger) {
  const options: echoScuOptions = {
    source: config.get(ConfParams.SOURCE),
    target,
    verbose: config.get(ConfParams.VERBOSE),
  };

  log.info(`sending C-ECHO to target: ${options.target?.aet}`);

  return new Promise((resolve, reject) => {
    echoScu(options, (result: string) => {
      if (result && result.length > 0) {
        try {
          log.info(`${options.target?.aet} replied to C-ECHO`);
          resolve(true);
        } catch (error) {
          log.error(result);
          reject(error);
        }
      }
      reject();
    });
  });
}
