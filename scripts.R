# Load the necessary library for reading CSV files
library(readr)

# Load the dataset
data <-clean_dataset

# Function to check if Q6 is valid
is_q6_valid <- function(q6_value) {
  if(is.na(q6_value)) return(FALSE)
  categories <- strsplit(as.character(q6_value), ",")[[1]]
  for(category in categories) {
    parts <- strsplit(category, "=>")[[1]]
    if(length(parts) != 2 || !grepl("^[0-9]+$", parts[2])) {
      return(FALSE)
    }
  }
  return(TRUE)
}

# Apply the function across Q6 and identify valid/invalid rows
data$Q6_valid <- sapply(data$Q6, is_q6_valid)

# Identifying rows with any missing values
rows_with_missing_values <- which(rowSums(is.na(data)) > 0)

# Combine indices of rows with missing values and invalid Q6
all_invalid_indices <- unique(c(rows_with_missing_values, which(!data$Q6_valid)))

# Data with missing values or invalid Q6
data_with_missing_values <- data[all_invalid_indices, ]

# Removing these rows from the original dataset to create a clean dataset
data_clean <- data[-all_invalid_indices, ]

# Remove the temporary Q6_valid column
data_clean$Q6_valid <- NULL

# Printing information
cat("Rows with missing values or invalid Q6:", length(all_invalid_indices), "\n")
cat("Data clean Rows:", nrow(data_clean), "Columns:", ncol(data_clean), "\n")

# Save data_with_missing_values to a CSV for analysis
write.csv("~/CSC311H5/ML challenge/ML_Challenge/output/data_with_missing_values.csv", "data_with_missing_values.csv", row.names = FALSE)

# Check for duplicate rows based on Q6 and remove them
rows_with_duplicates <- apply(data_clean, 1, function(row) {
  parts <- unlist(strsplit(as.character(row["Q6"]), ","))
  unique_values <- unique(as.numeric(gsub(".*=>", "", parts)))
  length(unique_values) != length(parts)
})

# Subset the data to include only rows with duplicate rankings
data_with_duplicates <- data_clean[rows_with_duplicates, ]
cat("Rows with duplicate rankings:", nrow(data_with_duplicates), "\n")

# Remove rows with duplicate rankings
data_clean <- data_clean[!rows_with_duplicates, ]

# Write data with duplicates removed to a CSV
write.csv(processed_data, file = "~/CSC311H5/ML challenge/ML_Challenge/output/processed_data.csv", row.names = FALSE)
write.csv(data_with_duplicates, file = "~/CSC311H5/ML challenge/ML_Challenge/output/data_with_duplicates.csv", row.names = FALSE)
write.csv(data_with_missing_values,"~/CSC311H5/ML challenge/ML_Challenge/output/data_with_missing_values.csv", row.names = FALSE)

# Display cleaned data information
cat("Cleaned data now have Row:", nrow(data_clean), ", Columns:", ncol(data_clean), "\n")
head(data_clean)

