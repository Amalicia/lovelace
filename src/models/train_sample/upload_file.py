from google.cloud import storage

def upload_file(bucket_name, source_file_name, destination_blob_name):
	print('Uploading file to cloud')
	storage_client = storage.Client()
	bucket = storage_client.bucket(bucket_name)
	blob = bucket.blob(destination_blob_name)

	blob.upload_from_filename(source_file_name)
	print('File uploaded')