% Define the folder path where the CSV files are located
folderPath = "/Users/hello/Desktop/Sazid Folder Arrangement/features file autism/Normal";

% Get a list of all CSV files in the folder
fileList = dir(fullfile(folderPath, '*.csv'));

% Initialize an empty table to hold all the data
allData = [];

% Loop through each file and read its content
for i = 1:length(fileList)
    % Construct the full file name
    fileName = fullfile(folderPath, fileList(i).name);
    
    % Read the current CSV file into a table
    tempData = readtable(fileName);
    
    % Append the data to the combined table
    allData = [allData; tempData];
end

% Optionally, save the merged data to a new CSV file
writetable(allData, 'Normalcsvfile.csv');
