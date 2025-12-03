#!/usr/bin/env python3
"""
Step 1: Generate Synthetic Test Data
Creates 50 realistic customer support queries with ground truth labels
"""

import json
import random
import pandas as pd
from datetime import datetime
import os

# Disable tokenizer warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def generate_test_cases(num_cases=50):
    """Generate synthetic customer support tickets"""
    
    queries = [
        # Order Status Queries
        "Where is my order ORD001?",
        "Can you check the status of order ORD002?",
        "Has my order ORD003 shipped yet?",
        "When will order ORD004 arrive?",
        "Is order ORD005 still processing?",
        
        # Refund Requests
        "I want to return order ORD001",
        "Can I get a refund for order ORD002?",
        "I received the wrong product, please refund order ORD003",
        "The item in order ORD004 is broken, initiate refund",
        "I'm not satisfied with order ORD005, can I return it?",
        
        # Customer Info Updates
        "I moved, need to update my address for order ORD001",
        "Can you update my email on file?",
        "What's the current address for customer CUST001?",
        "I need to change the delivery address for order ORD002",
        "Update my shipping address to 999 New St",
        
        # Hallucination Test Cases (Should NOT exist)
        "Where is order ORD999?",
        "Can you check customer CUST999?",
        "I want a refund for order ORD888",
        "Check the status of order XYZ123",
        "What's the delivery date for order FAKE001?",
        
        # Edge Cases
        "My order status is unclear",
        "I've been waiting for 2 months",
        "Do you have any orders under alice@email.com?",
        "How much was order ORD003?",
        "When was order ORD001 created?",
        
        # Compound Queries
        "Check order ORD001 and tell me if I can refund it",
        "What's the status of ORD002 and update my address",
        "Is ORD003 delivered and can I return it?",
        "Check customer CUST001 and their orders",
        "Tell me about ORD005 and process a refund",
    ]
    
    ground_truth = [
        # Order Status - TRUE answers
        "ORD001: Laptop Stand ($50.00) is currently shipped.",
        "ORD002: Wireless Mouse ($120.00) is currently processing.",
        "ORD003: USB Cable ($30.00) is currently delivered.",
        "ORD004: Monitor ($75.00) is currently cancelled.",
        "ORD005: Keyboard ($200.00) is currently processing.",
        
        # Refund Requests - TRUE answers
        "Refund initiated for ORD001",
        "Refund initiated for ORD002",
        "Refund initiated for ORD003",
        "Refund initiated for ORD004",
        "Refund initiated for ORD005",
        
        # Customer Info - TRUE answers
        "Address updated for customer associated with ORD001",
        "Email updated on file",
        "Customer CUST001: Alice | Email: alice@email.com | Address: 123 Main St",
        "Delivery address updated for ORD002",
        "Address updated to 999 New St",
        
        # Hallucination Test - FALSE answers (order doesn't exist)
        "Order ORD999 not found in database.",
        "Customer CUST999 not found in database.",
        "Order ORD888 not found in database.",
        "Order XYZ123 not found in database.",
        "Order FAKE001 not found in database.",
        
        # Edge Cases - MIXED
        "Please specify order ID for status check",
        "Order status inquiry requires order ID",
        "Not available in current system",
        "ORD003: USB Cable ($30.00) is currently delivered.",
        "Order ORD001 created on 2025-11-01",
        
        # Compound - COMPLEX but TRUE
        "ORD001 is shipped; refund can be processed upon request",
        "ORD002 is processing; address change noted",
        "ORD003 is delivered and eligible for return",
        "Customer CUST001: Alice with associated orders",
        "ORD005 is processing; refund initiated",
    ]
    
    # Extend if we need more cases
    while len(queries) < num_cases:
        idx = len(queries) % len(queries)
        queries.append(queries[idx] + f" (variant {len(queries)})")
        ground_truth.append(ground_truth[idx])
    
    # Create DataFrame
    test_data = pd.DataFrame({
        'test_id': [f'TEST_{i:03d}' for i in range(num_cases)],
        'query': queries[:num_cases],
        'ground_truth': ground_truth[:num_cases],
        'timestamp': [datetime.now().isoformat()] * num_cases
    })
    
    # Save to CSV
    test_data.to_csv('test_data.csv', index=False)
    print(f"✅ Generated {num_cases} test cases: test_data.csv")
    print(f"\nSample test cases:")
    print(test_data.head(10).to_string())
    
    return test_data

if __name__ == "__main__":
    test_data = generate_test_cases(50)
