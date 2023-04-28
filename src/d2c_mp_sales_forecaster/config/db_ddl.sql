CREATE TABLE streamlit.fcst_sessions (
  fcst_session_id SERIAL PRIMARY KEY,
  fcst_session_name TEXT UNIQUE,
  fcst_date DATE NOT NULL,
  user_tag TEXT
);

CREATE TABLE streamlit.product_fcst (
  fcst_session_id INTEGER NOT NULL REFERENCES streamlit.fcst_sessions(fcst_session_id),
  product TEXT NOT NULL,
  ds DATE NOT NULL,
  y FLOAT NOT NULL,
  yhat FLOAT NOT NULL,
  fcst_qty INTEGER NOT NULL,
  PRIMARY KEY (fcst_session_id, product, ds)
);

CREATE TABLE streamlit.product_size_fcst (
  fcst_session_id INTEGER NOT NULL REFERENCES streamlit.fcst_sessions(fcst_session_id),
  unique_id TEXT NOT NULL,
  ds DATE NOT NULL,
  yhat FLOAT NOT NULL,
  fcst_qty INTEGER NOT NULL,
  PRIMARY KEY (fcst_session_id, unique_id, ds)
);