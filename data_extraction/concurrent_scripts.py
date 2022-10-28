#!/bin/bash
#!/bin/sh

python data_extraction.py 2014 first &
python data_extraction.py 2014 second &
python data_extraction.py 2014 third &
python data_extraction.py 2014 fourth &