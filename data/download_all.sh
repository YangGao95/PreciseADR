for yr in {2013..2022}
do
  for qtr in {1..4}
  do
    curl -ko ./${yr}q${qtr}.zip "https://fis.fda.gov/content/Exports/faers_xml_${yr}q${qtr}.zip"
    mkdir ${yr}q${qtr}
    unzip ${yr}q${qtr}.zip -d ${yr}q${qtr}/
    rm ${yr}q${qtr}/xml/*.pdf
    rm ${yr}q${qtr}/xml/*.doc
    rm ${yr}q${qtr}/*.pdf
    rm ${yr}q${qtr}/*.doc
    rm -rf ${yr}q${qtr}/Deleted
  done
done


