% Define the path to your CSV file
filePath = "/Users/hello/Desktop/untitled folder 11/ASDcsvfile.csv";  % Replace with your actual file path

% Read the CSV file into a table
data = readtable(filePath);

% Find the column index for "Label"
labelColumnIdx = find(strcmp(data.Properties.VariableNames, 'Label'));

% Overwrite the "Label" column with 1
data{:, labelColumnIdx} = 1;

% Save the modified table back to a CSV file
writetable(data, 'ASDlabel.csv');
