-- Set this to an authorized reviewer ID before running; NULL returns no rows.
SET @reviewer_user_id = NULL;

SELECT
    p2.DicomID AS "DicomID",
    l.Disease AS "Disease",
    p2.Result AS "Result"
FROM
    Labels AS l
    LEFT JOIN (
        SELECT
            d.DicomID,
            rpl.Result,
            rpl.LabelsID
        FROM
            RadiologistPredictionsLabels AS rpl
            INNER JOIN RadiologistPredictions AS rp ON rpl.RadiologistPredictionsID = rp.RadiologistPredictionsID
            INNER JOIN Dicoms AS d ON rp.DicomID = d.DicomID
        WHERE
            d.IsDiscarded = 0
            AND rp.UserID = @reviewer_user_id
    ) AS p2 ON l.LabelsID = p2.LabelsID
WHERE
    l.LabelsID > 39
ORDER BY
    p2.DicomID ASC;