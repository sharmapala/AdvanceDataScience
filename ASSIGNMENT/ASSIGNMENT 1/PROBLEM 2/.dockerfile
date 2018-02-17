FROM continuumio/anaconda3
FROM python:3
RUN mkdir -p /usr/src/assignpart2
WORKDIR /usr/src/assignpart2

RUN pip install pandas
RUN pip install boto3
RUN pip install numpy
RUN pip install urllib3
RUN pip install bs4
RUN pip install lxml
RUN pip install matplotlib
Add ADSAssign1Part2Final.py /

ENV year 1
ENV accessKey 1
ENV secretKey 1
ENV Location 1

COPY ADSAssign1Part2Final.py /usr/src/assignpart2/

CMD ["sh","-c", "python /usr/src/assignpart2/ADSAssign1Part2Final.py --year ${year} --accessKey ${accessKey} --secretKey ${secretKey} --Location ${Location}"]

