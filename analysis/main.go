// Command analysis computes statistical properties of the
// encoder's feature vectors.
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/tweetenc"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	var dataPath string
	var encPath string
	var numSamples int
	var batchSize int
	flag.StringVar(&dataPath, "data", "", "tweet data")
	flag.StringVar(&encPath, "encoder", "../train/enc_out", "encoder network")
	flag.IntVar(&numSamples, "num", 512, "number of samples")
	flag.IntVar(&batchSize, "batch", 32, "batch size")
	flag.Parse()

	if dataPath == "" {
		fmt.Fprintln(os.Stderr, "Missing -data flag. See -help for more.")
		os.Exit(1)
	}

	log.Println("Loading encoder...")
	var encoder tweetenc.Encoder
	if err := serializer.LoadAny(encPath, &encoder); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to load encoder:", err)
		os.Exit(1)
	}

	log.Println("Loading samples...")
	samples, err := tweetenc.ReadSampleList(dataPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	anysgd.Shuffle(samples)

	log.Println("Computing statistics...")

	var mean, secondMoment anyvec.Vector
	var count int
	for i := 0; i < numSamples; i += batchSize {
		batch := samples.Slice(i, i+batchSize)
		var strs []string
		for j := 0; j < batch.Len(); j++ {
			strs = append(strs, string(batch.(tweetenc.SampleList)[j]))
		}
		outs := encoder.Encode(strs...)
		outsSq := outs.Copy()
		outsSq.Mul(outs)
		thisSum := anyvec.SumRows(outs, outs.Len()/batch.Len())
		thisSumSq := anyvec.SumRows(outsSq, outs.Len()/batch.Len())
		if mean == nil {
			mean = thisSum
			secondMoment = thisSumSq
		} else {
			mean.Add(thisSum)
			secondMoment.Add(thisSumSq)
		}
		count += batch.Len()
		log.Printf("Processed %d samples", count)
	}

	divisor := mean.Creator().MakeNumeric(1 / float64(count))
	mean.Scale(divisor)
	secondMoment.Scale(divisor)

	meanSq := mean.Copy()
	meanSq.Mul(mean)
	stddev := secondMoment.Copy()
	stddev.Sub(meanSq)
	anyvec.Pow(stddev, stddev.Creator().MakeNumeric(0.5))

	printStats(mean, stddev)
}

func printStats(mean, stddev anyvec.Vector) {
	for i := 0; i < mean.Len(); i++ {
		meanVal := anyvec.Sum(mean.Slice(i, i+1))
		stdVal := anyvec.Sum(stddev.Slice(i, i+1))
		fmt.Printf("%d\tmean=%.3f\tstddev=%.3f\n", i, meanVal, stdVal)
	}
}
