/* Add your custom scripts here */
/* Add your custom scripts here */
document.addEventListener('DOMContentLoaded', () => {
    const formFieldsContainer = document.getElementById('form-fields');
    const predictionForm = document.getElementById('prediction-form');
    const predictionResultDiv = document.getElementById('prediction-result');
    const resultText = document.getElementById('result-text');
    const probabilitiesText = document.getElementById('probabilities-text');
    const messageBox = document.getElementById('message-box');
    const messageContent = document.getElementById('message-content');

    // Define the order of features as seen in your screenshot
    // This order is crucial for the model's input
    const featureOrder = [
        "city",
        "city_development_index",
        "gender",
        "relevent_experience",
        "enrolled_university",
        "education_level",
        "major_discipline",
        "experience",
        "company_size",
        "company_type",
        "last_new_job",
        "training_hours"
    ];

    // Function to show messages
    function showMessage(message, type = 'info') {
        messageContent.textContent = message;
        messageBox.classList.remove('hidden');
        if (type === 'error') {
            messageBox.classList.remove('bg-blue-100', 'border-blue-400', 'text-blue-700');
            messageBox.classList.add('bg-red-100', 'border-red-400', 'text-red-700');
        } else {
            messageBox.classList.remove('bg-red-100', 'border-red-400', 'text-red-700');
            messageBox.classList.add('bg-blue-100', 'border-blue-400', 'text-blue-700');
        }
        // Automatically hide after 5 seconds
        setTimeout(() => {
            messageBox.classList.add('hidden');
        }, 5000);
    }

    // Function to create form elements dynamically
    function createFormElements() {
        featureOrder.forEach(featureName => {
            const formGroup = document.createElement('div');
            formGroup.className = 'form-group';

            const label = document.createElement('label');
            label.htmlFor = featureName;
            // Format feature name for display (e.g., "city_development_index" -> "City Development Index")
            label.textContent = featureName.replace(/_/g, ' ').replace(/\b\w/g, char => char.toUpperCase()) + ':';
            label.className = 'block text-gray-700 text-sm font-bold mb-2';

            formGroup.appendChild(label);

            // Check if the feature is categorical (present in encodedData)
            if (encodedData.hasOwnProperty(featureName)) {
                // Categorical feature: create a select dropdown
                const select = document.createElement('select');
                select.id = featureName;
                select.name = featureName;
                select.className = 'shadow appearance-none border rounded-lg w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent';

                // Get options from encodedData and sort them for better UX
                // Create a default "Select an option"
                const defaultOption = document.createElement('option');
                defaultOption.value = "";
                defaultOption.textContent = `Select ${label.textContent.replace(':', '')}`;
                defaultOption.disabled = true;
                defaultOption.selected = true;
                select.appendChild(defaultOption);

                const options = Object.keys(encodedData[featureName]).sort();
                options.forEach(optionText => {
                    const option = document.createElement('option');
                    option.value = optionText; // Send the original text value to backend
                    option.textContent = optionText;
                    select.appendChild(option);
                });
                formGroup.appendChild(select);
            } else {
                // Numerical feature: create a number input
                const input = document.createElement('input');
                input.type = 'number';
                input.id = featureName;
                input.name = featureName;
                input.className = 'shadow appearance-none border rounded-lg w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent';

                // Set default values and steps based on typical ranges
                if (featureName === 'city_development_index') {
                    input.step = 'any';
                    input.value = '0.7'; // Example default
                    input.min = '0'; // Assuming index is non-negative
                    input.max = '1'; // Assuming index is between 0 and 1
                } else if (featureName === 'training_hours') {
                    input.step = '1';
                    input.value = '50'; // Example default
                    input.min = '0'; // Cannot have negative training hours
                } else {
                    input.step = 'any';
                    input.value = '0'; // Generic default
                }
                formGroup.appendChild(input);
            }
            formFieldsContainer.appendChild(formGroup);
        });
    }

    // Call the function to create form elements on page load
    createFormElements();

    // Handle form submission
    predictionForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default form submission

        const formData = {};
        let isValid = true;
        featureOrder.forEach(featureName => {
            const element = document.getElementById(featureName);
            if (element) {
                // For select elements, ensure an option other than the default is selected
                if (element.tagName === 'SELECT' && element.value === "") {
                    showMessage(`Please select an option for ${element.labels[0].textContent.replace(':', '')}.`, 'error');
                    isValid = false;
                    return; // Skip to next iteration
                }
                formData[featureName] = element.value;
            }
        });

        if (!isValid) {
            return; // Stop submission if validation fails
        }

        showMessage('Making prediction...', 'info');
        predictionResultDiv.classList.add('hidden'); // Hide previous result

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Prediction failed');
            }

            const data = await response.json();

            // Display prediction result
            resultText.textContent = `Prediction: ${data.prediction === 1 ? 'Looking for a job change' : 'Not looking for job change'}`;
            probabilitiesText.textContent = `Probabilities: No Enrollment: ${data.probabilities[0].toFixed(4)}, Will Enroll: ${data.probabilities[1].toFixed(4)}`;
            predictionResultDiv.classList.remove('hidden');
            showMessage('Prediction successful!', 'info');

        } catch (error) {
            console.error('Error:', error);
            showMessage(`Error: ${error.message}`, 'error');
            predictionResultDiv.classList.add('hidden'); // Hide result on error
        }
    });
});
