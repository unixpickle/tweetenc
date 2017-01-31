// Command encode adds tweet encodings to a CSV file.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/tweetenc"
)

func main() {
	var dataFile string
	var outFile string
	var encFile string
	var batchSize int

	flag.StringVar(&dataFile, "data", "", "input CSV file")
	flag.StringVar(&outFile, "out", "out.csv", "output CSV file")
	flag.StringVar(&encFile, "encoder", "../train/enc_out", "encoder file")
	flag.IntVar(&batchSize, "batch", 8, "computation batch size")
	flag.Parse()

	if dataFile == "" {
		essentials.Die("Missing -data flag. See -help for more info.")
	}

	log.Println("Loading encoder...")
	var enc tweetenc.Encoder
	if err := serializer.LoadAny(encFile, &enc.Block); err != nil {
		essentials.Die("Load encoder:", err)
	}

	log.Println("Reading samples...")
	dataReader, err := os.Open(dataFile)
	if err != nil {
		essentials.Die("Open data:", err)
	}
	csvReader := csv.NewReader(dataReader)
	dataContents, err := csvReader.ReadAll()
	dataReader.Close()
	if err != nil {
		essentials.Die("Read data:", err)
	}
	dataPerm := rand.Perm(len(dataContents))

	log.Println("Opening output file...")
	dataWriter, err := os.Create(outFile)
	if err != nil {
		essentials.Die("Open output:", err)
	}
	defer dataWriter.Close()
	csvWriter := csv.NewWriter(dataWriter)

	log.Println("Encoding...")
	var numDone int
	for numDone < len(dataPerm) {
		var records [][]string
		var samples []string
		for i := numDone; i < len(dataPerm) && i < numDone+batchSize; i++ {
			rec := dataContents[dataPerm[i]]
			records = append(records, rec)
			samples = append(samples, rec[len(rec)-1])
		}
		encoded := enc.Encode(samples...)
		vecSize := encoded.Len() / len(records)
		components := encoded.Data().([]float32)
		for i, record := range records {
			for j := vecSize * i; j < vecSize*(i+1); j++ {
				record = append(record, fmt.Sprintf("%f", components[j]))
			}
			if err := csvWriter.Write(record); err != nil {
				essentials.Die("Write record:", err)
			}
		}
		csvWriter.Flush()
		if err := csvWriter.Error(); err != nil {
			essentials.Die("Flush writer:", err)
		}
		numDone += len(records)
		log.Printf("Encoded %d samples", numDone)
	}
}
