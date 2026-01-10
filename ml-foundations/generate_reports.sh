#!/bin/bash
# Automated report generation script using Papermill

echo "🚀 Generating Iris Species Reports..."

# Create reports directory if it doesn't exist
mkdir -p notebooks/reports

# Array of species names
species=("setosa" "versicolor" "virginica")

# Generate report for each species
for sp in "${species[@]}"; do
    echo "📊 Generating report for: $sp"
    uv run papermill \
        notebooks/03_template_report.ipynb \
        notebooks/reports/report_${sp}.ipynb \
        -p species_name "$sp" \
        -p min_samples 10
    echo "✅ Report for $sp completed"
    echo ""
done

echo "🎉 All reports generated successfully!"
echo "📁 Reports saved in: notebooks/reports/"
ls -lh notebooks/reports/
