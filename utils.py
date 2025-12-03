#!/usr/bin/env python3
"""
Step 0: Database & Utility Setup
Initialize SQLite database and helper functions
"""

import sqlite3
import os
from datetime import datetime

def setup_database():
    """Create orders.db with sample data"""
    conn = sqlite3.connect("orders.db")
    cursor = conn.cursor()
    
    # Drop existing tables for fresh start
    cursor.execute("DROP TABLE IF EXISTS orders")
    cursor.execute("DROP TABLE IF EXISTS customers")
    cursor.execute("DROP TABLE IF EXISTS refunds")
    
    # Create orders table
    cursor.execute("""
    CREATE TABLE orders (
        order_id TEXT PRIMARY KEY,
        customer_id TEXT,
        status TEXT,
        amount REAL,
        product TEXT,
        created_date TEXT
    )
    """)
    
    # Create customers table
    cursor.execute("""
    CREATE TABLE customers (
        customer_id TEXT PRIMARY KEY,
        name TEXT,
        email TEXT,
        address TEXT
    )
    """)
    
    # Create refunds table
    cursor.execute("""
    CREATE TABLE refunds (
        refund_id TEXT PRIMARY KEY,
        order_id TEXT,
        status TEXT,
        reason TEXT,
        FOREIGN KEY(order_id) REFERENCES orders(order_id)
    )
    """)
    
    # Insert sample data
    sample_orders = [
        ("ORD001", "CUST001", "shipped", 50.0, "Laptop Stand", "2025-11-01"),
        ("ORD002", "CUST002", "processing", 120.0, "Wireless Mouse", "2025-11-15"),
        ("ORD003", "CUST003", "delivered", 30.0, "USB Cable", "2025-11-05"),
        ("ORD004", "CUST001", "cancelled", 75.0, "Monitor", "2025-11-20"),
        ("ORD005", "CUST004", "processing", 200.0, "Keyboard", "2025-11-28"),
    ]
    cursor.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)",
        sample_orders
    )
    
    sample_customers = [
        ("CUST001", "Alice", "alice@email.com", "123 Main St"),
        ("CUST002", "Bob", "bob@email.com", "456 Oak Ave"),
        ("CUST003", "Charlie", "charlie@email.com", "789 Pine Rd"),
        ("CUST004", "Diana", "diana@email.com", "321 Elm St"),
    ]
    cursor.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?)",
        sample_customers
    )
    
    conn.commit()
    conn.close()
    print("✅ Database initialized: orders.db")

def query_order_status(order_id):
    """SQL Tool: Check order status"""
    conn = sqlite3.connect("orders.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT status, amount, product FROM orders WHERE order_id = ?",
        (order_id,)
    )
    result = cursor.fetchone()
    conn.close()
    
    if result:
        status, amount, product = result
        return f"Order {order_id}: {product} (${amount}) is currently {status}."
    return f"Order {order_id} not found in database."

def query_customer_details(customer_id):
    """SQL Tool: Get customer information"""
    conn = sqlite3.connect("orders.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, email, address FROM customers WHERE customer_id = ?",
        (customer_id,)
    )
    result = cursor.fetchone()
    conn.close()
    
    if result:
        name, email, address = result
        return f"Customer: {name} | Email: {email} | Address: {address}"
    return f"Customer {customer_id} not found."

def initiate_refund(order_id, reason):
    """SQL Tool: Create refund request"""
    conn = sqlite3.connect("orders.db")
    cursor = conn.cursor()
    
    # Check if order exists
    cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
    if not cursor.fetchone():
        conn.close()
        return f"Cannot refund: Order {order_id} not found."
    
    refund_id = f"REF-{order_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    cursor.execute(
        "INSERT INTO refunds VALUES (?, ?, ?, ?)",
        (refund_id, order_id, "pending", reason)
    )
    conn.commit()
    conn.close()
    return f"Refund created: {refund_id} (Status: Pending Review)"

def get_tools_description():
    """Return available tools for the agent"""
    return [
        {
            "name": "query_order_status",
            "description": "Check the status of a customer order (e.g., shipped, processing, delivered)",
            "params": ["order_id"]
        },
        {
            "name": "query_customer_details",
            "description": "Retrieve customer information (name, email, address)",
            "params": ["customer_id"]
        },
        {
            "name": "initiate_refund",
            "description": "Create a refund request for an order",
            "params": ["order_id", "reason"]
        }
    ]

if __name__ == "__main__":
    setup_database()
    print("Tools available:", get_tools_description())
