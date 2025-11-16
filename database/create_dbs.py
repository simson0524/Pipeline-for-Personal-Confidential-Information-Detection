# database/create_dbs.py

from psycopg2.extras import execute_values
import psycopg2


def get_connection(config):
    return psycopg2.connect(
        host=config['db']['host'],        # service name
        port=config['db']['port'],        # postgreSQL port
        dbname=config['db']['dbname'],    # db name
        user=config['db']['user'],        # user
        password=config['db']['password'] # password
    )

def create_nessesary_tables(conn):
    cursor = conn.cursor()

    # [personal/confidential_info_dictionary]
    # 개인/기밀정보 사전 스키마
    for table in ['personal_info_dictionary', 'confidential_info_dictionary']:
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                domain_id                        TEXT,
                span_token                       TEXT,
                z_score                          REAL,
                first_inserted_experiment_name   TEXT,
                insertion_counts                 INTEGER,
                deletion_counts                  INTEGER,
                PRIMARY KEY (domain_id, span_token)
            );
        """)

    # Master Table : [experiment]
    # 실험에 관한 총 개요
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS experiment (
            experiment_name                 TEXT PRIMARY KEY,
            previous_experiment_name        TEXT,
            is_pii                          TEXT,
            batch_size                      TEXT,
            num_epochs                      TEXT,
            learning_rate                   TEXT,
            train_dir                       TEXT,
            valid_dir                       TEXT,
            experiment_start_time           TEXT,
            experiment_end_time             TEXT,
            model_train_duration            TEXT,
            dictionary_matching_duration    TEXT,
            ner_regex_matching_duration     TEXT,
            model_validation_duration       TEXT,
            aug_sent_generation_duration    TEXT,
            aug_sent_auto_valid_duration    TEXT,
            aug_sent_manual_valid_duration  TEXT,
            total_sentence_counts           INTEGER,
            total_annotated_token_counts    INTEGER
        );
    """)

    # [model_train_performance, model_train_sent_dataset_log]
    # 2. 모델학습검증 프로세스에 관한 테이블
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS model_train_performance (
            experiment_name         TEXT,
            performed_epoch         INTEGER,
            start_time              TIMESTAMPTZ,
            end_time                TIMESTAMPTZ,
            model_weight_file_path  TEXT,
            train_loss              TEXT,
            valid_loss              TEXT,
            precision               REAL,
            recall                  REAL,
            f1                      REAL,
            confusion_matrix        JSONB,
            PRIMARY KEY (experiment_name, performed_epoch),
            FOREIGN KEY (experiment_name) REFERENCES experiment (experiment_name) ON DELETE CASCADE
        );
    """)
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS model_train_sent_dataset_log (
            experiment_name         TEXT,
            sentence_id             TEXT,
            sentence                TEXT,
            domain_id               TEXT,
            validated_epoch         INTEGER,
            span_token              TEXT,
            dataset_idx             TEXT,
            ground_truth            TEXT,
            prediction              TEXT,
            source_file_name        TEXT,
            sentence_sequence       TEXT,
            FOREIGN KEY (experiment_name) REFERENCES experiment (experiment_name) ON DELETE CASCADE
        );
    """)

    # [dictionary_matching_performance, dictionary_matching_sent_dataset_log]
    # 3. 사전매칭검증 프로세스에 관한 테이블
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS dictionary_matching_performance (
            experiment_name             TEXT PRIMARY KEY,
            start_time                  TIMESTAMPTZ,
            end_time                    TIMESTAMPTZ,
            hit_counts                  INTEGER,
            hit_delta_rate              REAL,
            wrong_counts                INTEGER,
            wrong_delta_rate            REAL,
            mismatch_counts             INTEGER,
            mismatch_delta_rate         REAL,
            dictionary_size             JSONB,
            dictionary_size_delta_rate  JSONB,
            confusion_matrix            JSONB,
            FOREIGN KEY (experiment_name) REFERENCES experiment (experiment_name) ON DELETE CASCADE
        );
    """)
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS dictionary_matching_sent_dataset_log (
            experiment_name         TEXT,
            sentence_id             TEXT,
            sentence                TEXT,
            domain_id               TEXT,
            span_token              TEXT,
            dataset_idx             TEXT,
            ground_truth            TEXT,
            prediction              TEXT,
            source_file_name        TEXT,
            sentence_sequence       TEXT,
            FOREIGN KEY (experiment_name) REFERENCES experiment (experiment_name) ON DELETE CASCADE
        );
    """)

    # [ner_regex_matching_performance, ner_regex_matching_sent_dataset_log]
    # 4. NER/REGEX매칭검증 프로세스에 관한 테이블
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS ner_regex_matching_performance (
            experiment_name             TEXT PRIMARY KEY,
            start_time                  TIMESTAMPTZ,
            end_time                    TIMESTAMPTZ,
            hit_counts                  INTEGER,
            hit_delta_rate              REAL,
            wrong_counts                INTEGER,
            wrong_delta_rate            REAL,
            mismatch_counts             INTEGER,
            mismatch_delta_rate         REAL,
            confusion_matrix            JSONB,
            FOREIGN KEY (experiment_name) REFERENCES experiment (experiment_name) ON DELETE CASCADE
        );
    """)
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS ner_regex_matching_sent_dataset_log (
            experiment_name         TEXT,
            sentence_id             TEXT,
            sentence                TEXT,
            domain_id               TEXT,
            span_token              TEXT,
            dataset_idx             TEXT,
            ground_truth            TEXT,
            prediction              TEXT,
            source_file_name        TEXT,
            sentence_sequence       TEXT,
            FOREIGN KEY (experiment_name) REFERENCES experiment (experiment_name) ON DELETE CASCADE
        );
    """)

    # [model_validation_performance, model_validation_sent_dataset_log]
    # 5. 모델검증 프로세스에 관한 테이블
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS model_validation_performance (
            experiment_name         TEXT PRIMARY KEY,
            start_time              TIMESTAMPTZ,
            end_time                TIMESTAMPTZ,
            model_weight_file_path  TEXT,
            best_performed_epoch    JSONB,
            precision               REAL,
            recall                  REAL,
            f1                      REAL,
            confusion_matrix        JSONB,
            FOREIGN KEY (experiment_name) REFERENCES experiment (experiment_name) ON DELETE CASCADE
        );
    """)
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS model_validation_sent_dataset_log (
            experiment_name         TEXT,
            sentence_id             TEXT,
            sentence                TEXT,
            domain_id               TEXT,
            span_token              TEXT,
            dataset_idx             TEXT,
            ground_truth            TEXT,
            prediction              TEXT,
            source_file_name        TEXT,
            sentence_sequence       TEXT,
            FOREIGN KEY (experiment_name) REFERENCES experiment (experiment_name) ON DELETE CASCADE
        );
    """)

    # [generation_augmented_performance, generation_augmented_sent_dataset_log]
    # 6. 문장증강 프로세스에 관한 테이블
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS generation_augmented_performance (
            experiment_name              TEXT PRIMARY KEY,
            start_time                   TIMESTAMPTZ,
            auto_validation_start_time   TIMESTAMPTZ,
            manual_validation_start_time TIMESTAMPTZ,
            end_time                     TIMESTAMPTZ,
            total_generation_counts      INTEGER,
            auto_validated_counts        INTEGER,
            manual_validated_counts      INTEGER,
            FOREIGN KEY (experiment_name) REFERENCES experiment (experiment_name) ON DELETE CASCADE
        );
    """)
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS generation_augmented_sent_dataset_log (
            sentence_id             TEXT PRIMARY KEY,
            experiment_name         TEXT,
            generated_sentence      TEXT,
            domain_id               TEXT,
            span_token              TEXT,
            target_ground_truth     TEXT,
            validated_label         TEXT,
            FOREIGN KEY (experiment_name) REFERENCES experiment (experiment_name) ON DELETE CASCADE
        );
    """)

    conn.commit()
    cursor.close()