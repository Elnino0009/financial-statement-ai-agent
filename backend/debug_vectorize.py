#!/usr/bin/env python3
"""
Vectorize.io Connection Diagnostic Tool
This script helps diagnose connection issues with Vectorize.io
"""

import os
import asyncio
import aiohttp
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def diagnose_vectorize_connection():
    """Comprehensive diagnosis of Vectorize.io connection"""
    
    print("🔍 Vectorize.io Connection Diagnostic")
    print("=" * 50)
    
    # Check environment variables
    api_key = os.getenv('VECTORIZE_API_KEY')
    org_id = os.getenv('VECTORIZE_ORG_ID')
    pipeline_id = os.getenv('VECTORIZE_PIPELINE_ID')
    
    print(f"✅ Environment Variables:")
    print(f"   VECTORIZE_API_KEY: {'✓' if api_key else '✗ Missing'}")
    print(f"   VECTORIZE_ORG_ID: {'✓' if org_id else '✗ Missing'} ({org_id[:8]}... if present)")
    print(f"   VECTORIZE_PIPELINE_ID: {'✓' if pipeline_id else '✗ Missing'} ({pipeline_id[:8]}... if present)")
    print()
    
    if not all([api_key, org_id, pipeline_id]):
        print("❌ Missing required environment variables. Please check your .env file.")
        return
    
    # Test different API endpoints
    endpoints_to_test = [
        # Current implementation
        f"https://api.vectorize.io/v1/organizations/{org_id}/pipelines/{pipeline_id}",
        # Alternative structures to test
        f"https://api.vectorize.io/v1/organizations/{org_id}",
        f"https://api.vectorize.io/v1/pipelines/{pipeline_id}",
        # Root endpoint
        "https://api.vectorize.io/v1/",
    ]
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    print("🌐 Testing API Endpoints:")
    print("-" * 30)
    
    async with aiohttp.ClientSession() as session:
        for i, url in enumerate(endpoints_to_test, 1):
            try:
                print(f"{i}. Testing: {url}")
                async with session.get(url, headers=headers) as response:
                    print(f"   Status: {response.status}")
                    
                    if response.status == 200:
                        try:
                            data = await response.json()
                            print(f"   ✅ SUCCESS! Response preview: {json.dumps(data, indent=2)[:200]}...")
                            return data
                        except:
                            text = await response.text()
                            print(f"   ✅ SUCCESS! Response: {text[:200]}...")
                            return text
                    else:
                        error_text = await response.text()
                        print(f"   ❌ Error: {error_text}")
                        
            except Exception as e:
                print(f"   ❌ Connection Error: {e}")
            
            print()
    
    # Test retrieval endpoint (the one used for searches)
    print("🔍 Testing Document Retrieval:")
    print("-" * 30)
    
    retrieval_url = f"https://api.vectorize.io/v1/organizations/{org_id}/pipelines/{pipeline_id}/retrieve"
    search_payload = {
        "question": "test query",
        "num_results": 1
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(retrieval_url, headers=headers, json=search_payload) as response:
                print(f"Retrieval Status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Document retrieval working! Found {len(data.get('documents', []))} documents")
                    if data.get('documents'):
                        print("Sample document keys:", list(data['documents'][0].keys()))
                else:
                    error_text = await response.text()
                    print(f"❌ Retrieval failed: {error_text}")
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
    
    print("\n📋 Recommendations:")
    print("-" * 20)
    print("1. Check your Vectorize.io dashboard for the correct Organization ID and Pipeline ID")
    print("2. Verify your API key has the correct permissions")
    print("3. Ensure your pipeline contains uploaded documents")
    print("4. Check Vectorize.io documentation for any API changes")

if __name__ == "__main__":
    asyncio.run(diagnose_vectorize_connection()) 