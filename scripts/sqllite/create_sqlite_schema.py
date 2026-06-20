import os
from sqlalchemy import create_engine, text

def build_sqlite_schema():
    """
    Builds a SQLite database schema for OLIST Data.
    """

    # Defining Output Path:
    output_dir= os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
    sqlite_path= os.path.join(output_dir, 'ecommerce_db.sqlite')

    # Removing existing file:
    if os.path.exists(sqlite_path):
        os.remove(sqlite_path)
        print("Existing SQLite file removed.")

    # Initializing SQLite Engine:
    sqlite_engine = create_engine(f"sqlite:///{sqlite_path}")

    # SQLite DDL for Table Creations:
    sqlite_ddl = f"""
        -- ==========================================
        -- LEVEL 1: PARENT TABLES
        -- ==========================================
        
        CREATE TABLE customers (
            customer_id VARCHAR(255) PRIMARY KEY,
            customer_unique_id VARCHAR(255),
            customer_zip_code_prefix INTEGER,
            customer_city VARCHAR(255),
            customer_state VARCHAR(2)
        );

        CREATE TABLE sellers (
            seller_id VARCHAR(255) PRIMARY KEY,
            seller_zip_code_prefix INTEGER,
            seller_city VARCHAR(255),
            seller_state VARCHAR(2)
        );

        CREATE TABLE products (
            product_id VARCHAR(255) PRIMARY KEY,
            product_category_name VARCHAR(255),
            product_name_lenght REAL,
            product_description_lenght REAL,
            product_photos_qty REAL,
            product_weight_g REAL,
            product_length_cm REAL,
            product_height_cm REAL,
            product_width_cm REAL
        );

        CREATE TABLE product_category_name_translation (
            product_category_name VARCHAR(255) PRIMARY KEY,
            product_category_name_english VARCHAR(255)
        );

        -- ==========================================
        -- LEVEL 2: CHILD TABLES (References Level 1)
        -- ==========================================

        CREATE TABLE orders (
            order_id VARCHAR(255) PRIMARY KEY,
            customer_id VARCHAR(255),
            order_status VARCHAR(50),
            order_purchase_timestamp DATETIME,
            order_approved_at DATETIME,
            order_delivered_carrier_date DATETIME,
            order_delivered_customer_date DATETIME,
            order_estimated_delivery_date DATETIME,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );

        -- ==========================================
        -- LEVEL 3: GRANDCHILD TABLES (References Level 1 & 2)
        -- ==========================================

        CREATE TABLE order_items (
            order_id VARCHAR(255),
            order_item_id INTEGER,
            product_id VARCHAR(255),
            seller_id VARCHAR(255),
            shipping_limit_date DATETIME,
            price REAL,
            freight_value REAL,
            PRIMARY KEY (order_id, order_item_id),
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id),
            FOREIGN KEY (seller_id) REFERENCES sellers(seller_id)
        );

        CREATE TABLE order_payments (
            order_id VARCHAR(255),
            payment_sequential INTEGER,
            payment_type VARCHAR(50),
            payment_installments INTEGER,
            payment_value REAL,
            PRIMARY KEY (order_id, payment_sequential),
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );

        CREATE TABLE order_reviews (
            review_id VARCHAR(255),
            order_id VARCHAR(255),
            review_score INTEGER,
            review_comment_title TEXT,
            review_comment_message TEXT,
            review_creation_date DATETIME,
            review_answer_timestamp DATETIME,
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );
    """

    # Executing DDL For Schema Creation:
    print('Building SQLite database schema...')
    with sqlite_engine.begin() as conn:
        for statement in sqlite_ddl.split(';'):
            if statement.strip():
                conn.execute(text(statement))

    print(f"SQLite Schema Successfully built at {sqlite_path}")

if __name__ == "__main__":
    build_sqlite_schema()