import { RowDataPacket } from "mysql2"

export interface IUsers extends RowDataPacket {
  UserID?: number
  Username: string
  Email: string
  Password: string
  IsAdmin: boolean
}

export interface IContacts extends RowDataPacket {
  Username?: number
  Email: string
  Message: string
  IsResolved: boolean
}

export interface IDicoms extends RowDataPacket {
  DicomID?: string
  AccessionNumber?: string
  StudyInstanceUID?: string
  SOPInstanceUID?: string
  SeriesInstanceUID?: string
  DiagnoseDate: Date
  InsertedAt: Date
  IsDiscarded: boolean
  DiscardingUserID: number
  DiscardedDate: Date,
  TimesLabeled: number
}

export interface IRadiologistPredictions extends RowDataPacket {
  radiologist_predictions_id?: number
  prediction_time: Date
  dicom_id: string
  user_id: number
}

export const diseaseIds = new Map<string, number>([
  ["/mass", 40],
  ["/mass/mediastinalmass", 41],
  ["/mass/breastmass", 42],
  ["/mass/pleuralmass", 43],
  ["/mass/pulmonarymass", 44],
  ["/mass/softtissuemass", 45],
  ["/lung", 46],
  ["/lung/urgent", 47],
  ["/lung/urgent/nodules", 48],
  ["/lung/urgent/pneumonia", 49],
  ["/lung/urgent/infiltrates", 50],
  ["/lung/urgent/infiltrates/interstitial", 51],
  ["/lung/urgent/infiltrates/alveolar", 52],
  ["/lung/urgent/atelectasis", 53],
  ["/lung/urgent/atelectasis/total", 54],
  ["/lung/urgent/atelectasis/lobar", 55],
  ["/lung/urgent/atelectasis/segmental", 56],
  ["/lung/urgent/atelectasis/laminar", 57],
  ["/lung/urgent/atelectasis/round", 58],
  ["/lung/urgent/atelectasis/basal", 59],
  ["/lung/epoc", 60],
  ["/lung/other", 61],
  ["/mediastinumhila", 62],
  ["/mediastinumhila/urgent", 63],
  ["/mediastinumhila/urgent/cardiomegaly", 64],
  ["/mediastinumhila/urgent/hila", 65],
  ["/mediastinumhila/urgent/hila/hilarcongestion", 66],
  ["/mediastinumhila/urgent/hila/hilarenlargement", 67],
  ["/mediastinumhila/urgent/trachealshift", 68],
  ["/mediastinumhila/vascularhilarenlargement", 69],
  ["/mediastinumhila/aorticelongation", 70],
  ["/mediastinumhila/aorticelongation/descendent", 71],
  ["/mediastinumhila/aorticelongation/ascendent", 72],
  ["/mediastinumhila/aorticelongation/button", 73],
  ["/mediastinumhila/aorticelongation/supraaortic", 74],
  ["/mediastinumhila/other", 75],
  ["/pdtw", 76],
  ["/pdtw/urgent", 77],
  ["/pdtw/urgent/pleuraleffusion", 78],
  ["/pdtw/urgent/pneumothorax", 79],
  ["/pdtw/urgent/costophrenicangleblunting", 80],
  ["/pdtw/urgent/fracture", 81],
  ["/pdtw/urgent/subcutaneousemphysema", 82],
  ["/pdtw/urgent/lyticblasticscleroticbonelesion", 83],
  ["/pdtw/incidental", 84],
  ["/pdtw/incidental/vertebraldegenerativechanges", 85],
  ["/pdtw/incidental/hemidiaphragmelevation", 86],
  ["/pdtw/incidental/callusribfracture", 87],
  ["/pdtw/incidental/nonaxialarticulardegenerativechanges", 88],
  ["/pdtw/incidental/chronicchanges", 89],
  ["/pdtw/hiatalhernia", 90],
  ["/pdtw/other", 91],
  ["/foreignbodies", 92],
  ["/foreignbodies/electricaldevices", 93],
  ["/foreignbodies/tube", 94],
  ["/foreignbodies/tube/nsgtube", 95],
  ["/foreignbodies/catheter", 96],
  ["/foreignbodies/surgery", 97],
  ["/calcification", 98],
  ["/pseudonodule", 99],
  ["/suboptimal", 100],
  ["/normal", 101],
]);
