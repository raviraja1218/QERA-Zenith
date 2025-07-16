# src/qera_core/data_fetchers/ibm_qiskit_data.py
from qiskit_ibm_provider import IBMProvider # Updated import for Qiskit 1.0+
from qiskit_aer.noise import NoiseModel

# IMPORTANT: NEVER COMMIT YOUR REAL IBM QUANTUM API TOKEN TO GITHUB!
# Store it securely (e.g., as a Colab Secret, or environment variable for local dev)
# For Colab, you can use: from google.colab import userdata; IBM_TOKEN = userdata.get('IBM_QUANTUM_TOKEN')
# For this setup, we will just use `IBMProvider()` to auto-load saved accounts.

# Attempt to load account (assumes token is saved, e.g., via IBMProvider.save_account())
# Or you can pass token directly: provider = IBMProvider(token='YOUR_API_TOKEN_HERE')
try:
    provider = IBMProvider() 
    print("IBM Quantum Provider initialized.")
except Exception as e:
    print(f"Could not initialize IBM Quantum Provider: {e}. Ensure account is loaded or token provided.")
    provider = None

def get_ibm_device_noise_model(device_name: str) -> NoiseModel:
    """
    Fetches the Qiskit Aer NoiseModel for a specified IBM Quantum device.
    Requires IBM Quantum account to be loaded/initialized.
    :param device_name: Name of the IBM Quantum device (e.g., 'ibm_lagos', 'ibm_washington').
                        Use recent, active devices.
    :return: A qiskit_aer.noise.NoiseModel object.
    :raises ValueError: If IBM Provider not initialized or device not found.
    """
    if provider is None:
        raise ValueError("IBM Quantum Provider not initialized. Cannot fetch device data.")

    try:
        # Get backend from provider
        backend = provider.get_backend(device_name)
        # Build noise model from backend properties and calibration
        noise_model = NoiseModel.from_backend(backend)
        print(f"Successfully loaded noise model for {device_name} from IBM Quantum.")
        return noise_model
    except Exception as e:
        raise ValueError(f"Could not get backend properties for {device_name}: {e}")

# Example usage for local/Colab testing
if __name__ == "__main__":
    # To make this work locally or in Colab:
    # 1. pip install qiskit-ibm-provider
    # 2. In your local Python environment: IBMProvider.save_account(token='YOUR_TOKEN')
    # 3. In Colab: add your token to Colab Secrets (left sidebar -> key icon), name it IBM_QUANTUM_TOKEN.
    #    Then: from google.colab import userdata; IBMProvider.save_account(token=userdata.get('IBM_QUANTUM_TOKEN'))

    print("--- Testing IBM Qiskit Data Fetcher ---")
    if provider is not None:
        try:
            # Use a recent, active IBM device for testing. Check IBM Quantum Lab for active devices.
            test_device = 'ibm_lagos' # Common 7-qubit device
            # test_device = 'ibm_sherbrooke' # Common 127-qubit device (if you have access)

            ibm_noise_model = get_ibm_device_noise_model(test_device)
            print(f"Fetched noise model summary for {test_device}:")
            print(ibm_noise_model)
            print(f"Number of errors in model: {len(ibm_noise_model._error_map)}")
        except ValueError as e:
            print(f"Error fetching IBM device noise model: {e}")
    else:
        print("Skipping IBM device data fetch example. Provider not initialized.")