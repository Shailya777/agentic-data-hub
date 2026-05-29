-- Login to MySQL DB as a root User and run following Script AFTER SUCCESSFUL DATA INGESTION (scripts/ingest_olist_data.py)

USE ecommerce_db;

-- 1. Convert TEXT IDs to VARCHAR so they can be indexed as Keys
-- Core Tables
ALTER TABLE customers 
MODIFY customer_id VARCHAR(255);

ALTER TABLE orders 
    MODIFY order_id VARCHAR(255), 
    MODIFY customer_id VARCHAR(255);

ALTER TABLE products 
    MODIFY product_id VARCHAR(255),
    MODIFY product_category_name VARCHAR(255);

ALTER TABLE order_items 
    MODIFY order_id VARCHAR(255), 
    MODIFY product_id VARCHAR(255),
    MODIFY seller_id VARCHAR(255);

ALTER TABLE sellers 
MODIFY seller_id VARCHAR(255);

ALTER TABLE order_payments 
MODIFY order_id VARCHAR(255);

ALTER TABLE product_category_name_translation 
MODIFY product_category_name VARCHAR(255);

ALTER TABLE order_reviews MODIFY order_id VARCHAR(255);


-- 2. Establish Primary Keys
-- Single Column Keys
ALTER TABLE customers ADD PRIMARY KEY (customer_id);
ALTER TABLE orders ADD PRIMARY KEY (order_id);
ALTER TABLE products ADD PRIMARY KEY (product_id);
ALTER TABLE sellers ADD PRIMARY KEY (seller_id);
ALTER TABLE product_category_name_translation ADD PRIMARY KEY (product_category_name);
-- Composite Keys (Because an order can have multiple items or multiple payment installments)
ALTER TABLE order_items ADD PRIMARY KEY (order_id, order_item_id);
ALTER TABLE order_payments ADD PRIMARY KEY (order_id, payment_sequential);


-- 3. Establish Foreign Keys
ALTER TABLE orders 
    ADD CONSTRAINT fk_orders_customers 
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id);

ALTER TABLE order_items 
    ADD CONSTRAINT fk_items_orders 
    FOREIGN KEY (order_id) REFERENCES orders(order_id);

ALTER TABLE order_items 
    ADD CONSTRAINT fk_items_products 
    FOREIGN KEY (product_id) REFERENCES products(product_id);

ALTER TABLE order_items
    ADD CONSTRAINT fk_items_sellers
    FOREIGN KEY (seller_id) REFERENCES sellers(seller_id);

ALTER TABLE order_payments
    ADD CONSTRAINT fk_payments_orders
    FOREIGN KEY (order_id) REFERENCES orders(order_id);

  -- Make sure the reviews table has an index for faster lookups
ALTER TABLE order_reviews ADD INDEX (order_id);

-- Add the Foreign Key linking it to the orders table
ALTER TABLE order_reviews
ADD CONSTRAINT fk_reviews_orders
FOREIGN KEY (order_id) REFERENCES orders(order_id);