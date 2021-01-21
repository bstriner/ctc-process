import os

from tensorflow.python.lib.io.tf_record import TFRecordCompressionType, TFRecordOptions, TFRecordWriter

COMPRESSION_TYPE = TFRecordCompressionType.GZIP
COMPRESSION_TYPE_STR = "GZIP"
class ShardRecordWriter(object):
    def __init__(self, path_fmt, chunksize, compression_type=COMPRESSION_TYPE):
        self.path_fmt = path_fmt
        self.chunksize = chunksize
        self.writer = None
        self.chunks = 0
        self.items = 0
        self.options = TFRecordOptions(compression_type=compression_type)
        # writer = tf.python_io.TFRecordWriter(outFilePath, options=options)

    def __enter__(self):
        self.open_writer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_writer()

    def output_file(self):
        return self.path_fmt.format(self.chunks)

    def open_writer(self):
        output_file = self.output_file()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        self.writer = TFRecordWriter(output_file, options=self.options)

    def close_writer(self):
        self.writer.close()
        self.writer = None

    def write(self, record):
        assert self.writer is not None
        if self.items >= self.chunksize:
            self.close_writer()
            self.items = 0
            self.chunks += 1
            self.open_writer()
        self.writer.write(record)
        self.items += 1
