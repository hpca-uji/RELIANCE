
export interface IDataBaseConfiguration {
    host: string;
    port: number;
    user: string;
    password: string;
    database: string;
}

export interface IConfigurationPeers {
    aet: string;
    ip: string;
    port: number;
    db: IDataBaseConfiguration;
  }