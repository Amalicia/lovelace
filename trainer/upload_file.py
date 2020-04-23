from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.cloud import storage
import logging

log = logging.getLogger(__name__)


# function to upload a file to cloud storage
def upload_file(bucket_name, source_file_name, destination_blob_name):
	log.info('Uploading file to cloud')
	# create client
	storage_client = storage.Client()
	# define bucket file will be uploaded to
	bucket = storage_client.bucket(bucket_name)

	# create blob
	blob = bucket.blob(destination_blob_name)

	# upload blob
	blob.upload_from_filename(source_file_name)
	log.info('File uploaded')
