terraform {
  backend "gcs" {
    bucket = "qwiklabs-gcp-03-faa5fcdada33-terraform-state"
    prefix = "dev"
  }
}
