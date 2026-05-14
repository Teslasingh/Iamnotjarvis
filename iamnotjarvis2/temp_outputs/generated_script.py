def main():
    try:
        print("Starting execution...", flush=True)

        # Get user inputs
        print("Enter number: ", end="", flush=True)
        num1 = float(input())  # Separate input line

        print("Enter number: ", end="", flush=True)
        num2 = float(input())  # Separate input line

        # Process data and show results
        result = num1 + num2
        new_result = result - 2
        print(f"Result: {new_result}", flush=True)

    except Exception as exc:
        print(f"Error: {exc}", flush=True)
        raise  # Required
    finally:
        print("Successfully", flush=True)  # Required

if __name__ == "__main__":
    main()