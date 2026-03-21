#!/usr/bin/env Rscript

# Install required packages if missing
if(!require(rmarkdown)) {
  install.packages("rmarkdown", repos="https://cloud.r-project.org")
}
if(!require(tinytex)) {
  install.packages("tinytex", repos="https://cloud.r-project.org")
}

# Check if TinyTeX is installed, if not install it
if(!tinytex::is_tinytex()) {
  cat("Installing TinyTeX (LaTeX distribution for R)...\n")
  cat("This may take a few minutes...\n")
  tinytex::install_tinytex()
}

# Knit the document to PDF
cat("Knitting GLM_Provisional_Analysis_Results.Rmd to PDF...\n")
rmarkdown::render(
  "GLM_Provisional_Analysis_Results.Rmd",
  output_format = "pdf_document",
  output_file = "GLM_Provisional_Analysis_Results.pdf"
)

cat("\n✓ PDF generated: GLM_Provisional_Analysis_Results.pdf\n")
