import { OkPacket, FieldPacket } from "mysql2";
import { IUsers, IDicoms } from "./tables";
import { FastifyInstance } from 'fastify';

export class UsersRepository {
    static async readAll(server: FastifyInstance): Promise<IUsers[]> {
        return new Promise((resolve, reject) => {
            server.mysql.query<IUsers[]>(
                "SELECT * FROM Users;",
            ).then(([rows, fields]: [IUsers[], FieldPacket[]]) => resolve(rows),
                (err) => reject(err))
                .catch((err) => reject(err));
        });
    }

    static async readById(server: FastifyInstance, userId: number): Promise<IUsers> {
        return new Promise<IUsers>((resolve, reject) => {
            server.mysql.execute<IUsers[]>(
                "SELECT * FROM Users WHERE UserID = ?;",
                [userId])
                .then(([rows, fields]: [IUsers[], FieldPacket[]]) => resolve(rows[0]),
                    (err) => reject(err))
                .catch((err) => reject(err));
        });
    }

    static async readByUser(server: FastifyInstance, username: string): Promise<IUsers> {
        return new Promise<IUsers>((resolve, reject) => {
            server.mysql.execute<IUsers[]>(
                "SELECT * FROM Users WHERE Username = ?;",
                [username])
                .then(([rows, fields]: [IUsers[], FieldPacket[]]) => resolve(rows[0]),
                    (err) => reject(err))
                .catch((err) => reject(err));
        });
    }

    static async create(server: FastifyInstance, user: IUsers): Promise<IUsers> {
        return new Promise<IUsers>((resolve, reject) => {
            server.mysql.execute<OkPacket>(
                "INSERT INTO Users (Username, Email, Password, isAdmin) VALUES(?,?,?,?)",
                [user.username, user.email, user.password, user.is_admin])
                .then(([rows, fields]: [OkPacket, FieldPacket[]]) => {
                    this.readById(server, rows.insertId).then((value: IUsers) => resolve(value), (err) => reject(err))
                }, (err) => reject(err)).catch((err) => reject(err));
        });
    }

    static async update(server: FastifyInstance, user: IUsers): Promise<IUsers> {
        return new Promise<IUsers>((resolve, reject) => {
            server.mysql.execute<OkPacket>(
                "UPDATE Users SET Username = ?, Email = ?, Password = ?, IsAdmin = ? WHERE UserID = ?",
                [user.Username, user.Email, user.Password, user.IsAdmin, user.UserID])
                .then(([rows, fields]: [OkPacket, FieldPacket[]]) => {
                    this.readById(server, rows.insertId).then((value: IUsers) => resolve(value), (err) => reject(err))
                }, (err) => reject(err)).catch((err) => reject(err));
        });
    }

    static async remove(server: FastifyInstance, userId: number): Promise<number> {
        return new Promise<number>((resolve, reject) => {
            server.mysql.execute<OkPacket>(
                "DELETE FROM Users WHERE UserID = ?",
                [userId])
                .then(([rows, fields]: [OkPacket, FieldPacket[]]) => resolve(rows.affectedRows), (err) => reject(err)).catch((err) => reject(err));
        });
    }
}

export class ContactsRepository {
    static async create(server: FastifyInstance, username: string, email: string, message: string): Promise<{ isCreated: boolean, err: string }> {
        return new Promise((resolve, reject) => {
            server.mysql.execute<OkPacket>(
                "INSERT INTO Contacts (Username, Email, Message, IsResolved) VALUES(?,?,?,?)",
                [username, email, message, false])
                .then(([rows, fields]: [OkPacket, FieldPacket[]]) => resolve({ isCreated: true, err: '' }),
                    (err) => resolve({ isCreated: false, err: err.toString() }))
                .catch((err) => resolve({ isCreated: false, err: err.toString() }));
        })
    }
}

interface IOHIFFilters {
    limit: number | undefined;
    offset: number | undefined;
    accessionNumber: string | undefined;
    studyInstanceUID: string | undefined;
    seriesInstanceUID: string | undefined;
    sopInstanceUID: string | undefined;
    receivedTime: { startDate: string, endDate: string } | undefined;
    predictionTime: { startDate: string, endDate: string } | undefined;
    timesLabeled: number | undefined;
}
interface IDBWhereQueryFilters {
    accessionFilter: string;
    studyFilter: string;
    seriesFilter: string;
    sopFilter: string;
    startDateFilter: string;
    finishDateFilter: string;
    receivedTimeFilter: string;
    predictionTimeFilter: string;
}
interface IDBHavingQueryFilters {
    timesLabeledFilter: string;
}
function buildDateQuery(dateFilter: { startDate: string, endDate: string } | undefined, dbRow: string): string {
    if (typeof dateFilter === 'undefined') return '';

    const { startDate, endDate } = dateFilter;

    if (endDate < startDate) {
        return `AND ${dbRow} >= '${startDate}'`;
    }
    return `AND ${dbRow} >= '${startDate}' AND ${dbRow} <= '${endDate}'`;
}
function buildWhereQueryFromFilters(filters: IOHIFFilters): IDBWhereQueryFilters {
    return {
        accessionFilter: typeof filters.accessionNumber === 'undefined' ? "" : `AND d.AccessionNumber LIKE '${filters.accessionNumber.replace(/\*/g, '%')}'`,
        studyFilter: typeof filters.studyInstanceUID === 'undefined' ? "" : `AND d.StudyInstanceUID = '${filters.studyInstanceUID}'`,
        seriesFilter: typeof filters.seriesInstanceUID === 'undefined' ? "" : `AND d.SeriesInstanceUID = '${filters.seriesInstanceUID}'`,
        sopFilter: typeof filters.sopInstanceUID === 'undefined' ? "" : `AND d.SOPInstanceUID = '${filters.sopInstanceUID}'`,
        startDateFilter: `AND tp.ReceivedTime >= '2023-04-30 23:59'`,
        finishDateFilter: `AND tp.ReceivedTime <= '2023-10-31 23:59'`,
        receivedTimeFilter: buildDateQuery(filters.receivedTime, 'tp.ReceivedTime'),
        predictionTimeFilter: buildDateQuery(filters.predictionTime, 'rp.PredictionTime'),
    }
}
function buildHavingQueryFromFilters(filters: IOHIFFilters): IDBHavingQueryFilters {
    return {
        timesLabeledFilter: typeof filters.timesLabeled === 'undefined' ? "" : `HAVING TimesLabeled = ${filters.timesLabeled}`,
    }
}
export class DicomsRepository {
    static async readAll(server: FastifyInstance): Promise<IDicoms[]> {
        return new Promise((resolve, reject) => {
            server.mysql.query<IDicoms[]>(
                "SELECT * FROM Dicoms")
                .then(([rows, fields]: [IDicoms[], FieldPacket[]]) => resolve(rows), (err) => reject(err))
                .catch((err) => reject(err));
        });
    }

    static async readAllPendingOrdered(server: FastifyInstance, userId: string, filters: IOHIFFilters): Promise<IDicoms[]> {
        const { limit, offset } = filters;
        const dbWhereFilters: IDBWhereQueryFilters = buildWhereQueryFromFilters(filters);
        const dbHavingFilters: IDBHavingQueryFilters = buildHavingQueryFromFilters(filters);
        const readAllPendingToUser = `
            SELECT DISTINCT 
                d.AccessionNumber AS 'AccessionNumber',
                d.StudyInstanceUID AS 'StudyInstanceUID',
                d.SOPInstanceUID AS 'SOPInstanceUID',
                d.SeriesInstanceUID AS 'SeriesInstanceUID',
                MAX(CASE WHEN (u.UserID != ? OR u.UserID IS NULL) THEN NULL ELSE rp.PredictionTime END) AS 'DiagnoseDate',
                MIN(tp.ReceivedTime) AS 'InsertedAt',
                COUNT(rp.RadiologistPredictionsID) AS 'TimesLabeled',
                MAX(CASE WHEN (u.UserID != ? OR u.UserID IS NULL) THEN 0 ELSE 1 END) AS 'DiagnosedByMe'
            FROM 
                Dicoms AS d 
                INNER JOIN ChestPredictions AS tp ON d.DicomID = tp.DicomID 
                LEFT JOIN RadiologistPredictions AS rp ON d.DicomID = rp.DicomID 
                LEFT JOIN Users AS u ON u.UserID = rp.UserID
            WHERE 
                d.StudyInstanceUID IS NOT NULL 
                AND d.SOPInstanceUID IS NOT NULL 
                AND d.SeriesInstanceUID IS NOT NULL 
                AND d.IsDiscarded = 0
                ${Object.values(dbWhereFilters).join(" ")}
            GROUP BY
                d.AccessionNumber,
                d.StudyInstanceUID,
                d.SOPInstanceUID,
                d.SeriesInstanceUID
            ${Object.values(dbHavingFilters).join(" ")}
            ORDER BY 
                ${filters.predictionTime === undefined ? "DiagnosedByMe ASC" : "DiagnosedByMe DESC"},
                InsertedAt DESC,
                DiagnoseDate DESC,
                TimesLabeled DESC
            LIMIT ?
            OFFSET ?;
        `;
        const sqlString = server.mysql.format(readAllPendingToUser, [userId, userId, limit, offset]);

        return new Promise((resolve, reject) => {
            server.mysql.execute<IDicoms[]>(
                sqlString)
                .then(([rows, fields]: [IDicoms[], FieldPacket[]]) => resolve(rows), (err) => reject(err))
                .catch((err) => reject(err));
        });
    }

    static async readNotDiagnosed(server: FastifyInstance, userId: number): Promise<IDicoms[]> {
        const queryString = `
            SELECT d.DicomID
            FROM Dicoms AS d LEFT JOIN RadiologistPredictions AS rp ON d.DicomID = rp.DicomID
            WHERE rp.UserID != ?
        `;
        const sqlString = server.mysql.format(queryString, [userId]);

        return new Promise((resolve, reject) => {
            server.mysql.execute<IDicoms[]>(
                sqlString)
                .then(([rows, fields]: [IDicoms[], FieldPacket[]]) => resolve(rows), (err) => reject(err))
                .catch((err) => reject(err));
        });
    }

    static async readId(server: FastifyInstance, studyInstanceUid: string, seriesInstanceUid: string, sopInstanceUid: string): Promise<string | undefined> {
        const queryString = `
            SELECT d.DicomID
            FROM Dicoms AS d
            WHERE d.StudyInstanceUID = ? AND d.SeriesInstanceUid = ? AND d.SOPInstanceUid = ?
        `;
        const sqlString = server.mysql.format(queryString, [studyInstanceUid, seriesInstanceUid, sopInstanceUid]);

        return new Promise((resolve, reject) => {
            server.mysql.execute<IDicoms[]>(
                sqlString)
                .then(([rows, fields]: [IDicoms[], FieldPacket[]]) => resolve(rows[0].DicomID), (err) => reject(err))
                .catch((err) => reject(err));
        });
    }

    static async readIdByStudyInstanceUID(server: FastifyInstance, studyInstanceUid: string): Promise<string | undefined> {
        const queryString = `
            SELECT d.DicomID
            FROM Dicoms AS d
            WHERE d.StudyInstanceUID = ?
        `;
        const sqlString = server.mysql.format(queryString, [studyInstanceUid]);

        return new Promise((resolve, reject) => {
            server.mysql.execute<IDicoms[]>(
                sqlString)
                .then(([rows, fields]: [IDicoms[], FieldPacket[]]) => resolve(rows[0].DicomID), (err) => reject(err))
                .catch((err) => reject(err));
        });
    }

    static async remove(server: FastifyInstance, dicomId: number): Promise<number> {
        return new Promise((resolve, reject) => {
            server.mysql.execute<OkPacket>(
                "DELETE FROM Dicoms WHERE DicomID = ?",
                [dicomId])
                .then(([rows, fields]: [OkPacket, FieldPacket[]]) => resolve(rows.affectedRows), (err) => reject(err))
                .catch((err) => reject(err));
        })
    }

    static async discard(server: FastifyInstance, dicomId: string, userId: string): Promise<boolean> {
        return new Promise((resolve, reject) => {
            server.mysql.execute<OkPacket>(
                "UPDATE Dicoms SET IsDiscarded = ?, DiscardingUserID = ?, DiscardedDate = ? WHERE DicomID = ?",
                [true, userId, new Date(), dicomId])
                .then(([rows, fields]: [OkPacket, FieldPacket[]]) => resolve(true), (err) => reject(err))
                .catch((err) => reject(err));
        })
    }
}

export class RadiologistPredictionsRepository {
    static async insertPredictionOld(server: FastifyInstance, userId: string, dicomId: string, labelsIds: (number | undefined)[]): Promise<boolean> {
        return new Promise((resolve, reject) => {
            server.mysql.execute<OkPacket>(
                "INSERT INTO RadiologistPredictions (PredictionTime, DicomID, UserID) VALUES(?,?,?)",
                [new Date(), dicomId, userId])
                .then(([rows, fields]: [OkPacket, FieldPacket[]]) => {
                    const insertion_results: boolean[] = new Array(labelsIds.keys.length);
                    Object.values(labelsIds).forEach((labelsId: number | undefined) => {
                        if (labelsId === undefined) return insertion_results.push(false);
                        this.insertLabel(server, rows.insertId, labelsId, 1)
                            .then(answer => insertion_results.push(answer))
                            .catch(reject);
                    });
                    resolve(insertion_results.every(v => v))
                }, (err) => reject(err)).catch(err => reject(err));
        });
    }
    static async insertPrediction(server: FastifyInstance, userId: string, dicomId: string, labelsIds: (number | undefined)[]): Promise<boolean> {
        const [rows] = await server.mysql.execute<OkPacket>(
            "INSERT INTO RadiologistPredictions (PredictionTime, DicomID, UserID) VALUES(?,?,?)",
            [new Date(), dicomId, userId]
        );

        const insertionResults = await Promise.all(
            labelsIds
                .filter((labelsId): labelsId is number => labelsId !== undefined)
                .map(labelsId => this.insertLabel(server, rows.insertId, labelsId, 1))
        );

        return insertionResults.every(result => result);
    }


    static async insertLabel(server: FastifyInstance, radiologistId: number, key: number, value: number): Promise<boolean> {
        return new Promise((resolve, reject) => {
            server.mysql.execute<OkPacket>(
                "INSERT INTO RadiologistPredictionsLabels (RadiologistPredictionsID, LabelsID, result) VALUES(?,?,?)",
                [radiologistId, key, value])
                .then(([rows, fields]: [OkPacket, FieldPacket[]]) => resolve(true), (err) => reject(err))
                .catch((err) => reject(err));
        });
    }
}
