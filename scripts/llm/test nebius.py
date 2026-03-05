import asyncio
import aiohttp
import json
import time


HUGGINGFACE_URL = "https://api.studio.nebius.com/v1/chat/completions"
HUGGINGFACE_API_KEY = ''

async def test_nebius_request():
    # Replace with your actual API key
    api_key = HUGGINGFACE_API_KEY

    # Full endpoint URL including the path
    api_url = HUGGINGFACE_URL

    # Request payload
    data_dict = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "user", "content": "Hello, this is a test. Please respond briefly."}
        ],
        "temperature": 0.5
    }

    # Convert to JSON string
    data = json.dumps(data_dict)

    # Create session and make request
    async with aiohttp.ClientSession() as session:
        try:
            start_time = time.time()

            async with session.post(
                    api_url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=data_dict
            ) as response:
                # Print status for diagnostics
                print(f"Status: {response.status}")
                print(f"Headers: {response.headers}")

                # Check if request was successful
                if response.status == 200:
                    # Parse and print response
                    parsed_response = await response.json()
                    end_time = time.time()
                    processing_time = end_time - start_time

                    print(f"Response received in {processing_time:.2f} seconds:")
                    print(f"Content: {parsed_response['choices'][0]['message']['content']}")
                    print("\nFull response:")
                    print(json.dumps(parsed_response, indent=2))
                else:
                    # Get error details
                    error_text = await response.text()
                    print(f"Error response: {error_text}")

        except Exception as e:
            print(f"Exception occurred: {type(e).__name__}: {e}")


async def test_json_response():
    # Replace with your actual API key
    # Replace with your actual API key
    api_key = HUGGINGFACE_API_KEY

    # Full endpoint URL including the path
    api_url = HUGGINGFACE_URL

    # Request with JSON response format
    data_dict = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "user", "content": "List three colors in JSON format with their hex codes."}
        ],
        "temperature": 0.5,
        "response_format": {"type": "json_object"}
    }

    data = json.dumps(data_dict)

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                    api_url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=data_dict
            ) as response:
                print(f"\nJSON Test Status: {response.status}")

                if response.status == 200:
                    parsed_response = await response.json()
                    json_content = parsed_response['choices'][0]['message']['content']

                    print("JSON response received:")
                    print(json_content)

                    # Verify JSON is valid
                    try:
                        json_data = json.loads(json_content)
                        print("Successfully parsed JSON response!")
                    except json.JSONDecodeError:
                        print("WARNING: Response is not valid JSON")
                else:
                    error_text = await response.text()
                    print(f"JSON test error response: {error_text}")

        except Exception as e:
            print(f"JSON test exception: {type(e).__name__}: {e}")


# Run both tests
async def main():
    print("Testing basic API connectivity...")
    await test_nebius_request()

    print("\nTesting JSON response format...")
    await test_json_response()


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())