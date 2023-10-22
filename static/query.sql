CREATE TABLE Records (
    id INT PRIMARY KEY IDENTITY(1,1),
    country NVARCHAR(255),
    race NVARCHAR(255),
    hospital NVARCHAR(255),
    dyspnea INT,
    dm INT,
    spo2 INT,
    rr INT,
    crp FLOAT,
    ldh FLOAT,
    anc FLOAT,
    wbc FLOAT,
    alc FLOAT,
    plt FLOAT,
    prediction_proba FLOAT,
    actual_outcome NVARCHAR(255)
);