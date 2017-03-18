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
	var encoder *tweetenc.Encoder
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

	var centers, logStddevs []anyvec.Vector
	var batchSizes []int

	var count int
	for i := 0; i < numSamples; i += batchSize {
		batch := samples.Slice(i, i+batchSize)
		var strs []string
		for j := 0; j < batch.Len(); j++ {
			strs = append(strs, string(batch.(tweetenc.SampleList)[j]))
		}
		c, l := evalSeqs(encoder, strs)
		centers = append(centers, c)
		logStddevs = append(logStddevs, l)
		batchSizes = append(batchSizes, batch.Len())
		count += batch.Len()
		log.Printf("Processed %d samples", count)
	}

	centerMean, centerStddev := stats(centers, batchSizes)
	lsMean, lsStddev := stats(logStddevs, batchSizes)

	printStats(centerMean, centerStddev, lsMean, lsStddev)
}

func printStats(centerMean, centerStd, lsMean, lsStd anyvec.Vector) {
	for i := 0; i < centerMean.Len(); i++ {
		meanVal := anyvec.Sum(centerMean.Slice(i, i+1))
		stdVal := anyvec.Sum(centerStd.Slice(i, i+1))
		lsMean := anyvec.Sum(lsMean.Slice(i, i+1))
		lsStd := anyvec.Sum(lsStd.Slice(i, i+1))
		fmt.Printf("%d\tE[μ]=%.3f\tσ(μ)=%.3f\tE[ln(σ)]=%.3f\tσ(ln(σ))=%.3f\n",
			i, meanVal, stdVal, lsMean, lsStd)
	}
}

func stats(vals []anyvec.Vector, batches []int) (mean, stddev anyvec.Vector) {
	var moment1, moment2 anyvec.Vector
	var count int
	for i, x := range vals {
		x2 := x.Copy()
		x2.Mul(x)
		xSum := anyvec.SumRows(x, x.Len()/batches[i])
		x2Sum := anyvec.SumRows(x2, x2.Len()/batches[i])
		if moment1 == nil {
			moment1 = xSum
			moment2 = x2Sum
		} else {
			moment1.Add(xSum)
			moment2.Add(x2Sum)
		}
		count += batches[i]
	}

	divisor := moment1.Creator().MakeNumeric(1 / float64(count))
	moment1.Scale(divisor)
	moment2.Scale(divisor)

	meanSq := moment1.Copy()
	meanSq.Mul(moment1)
	stddev = moment2.Copy()
	stddev.Sub(meanSq)
	anyvec.Pow(stddev, stddev.Creator().MakeNumeric(0.5))

	return moment1, stddev
}

func evalSeqs(e *tweetenc.Encoder, samples []string) (center, logStddev anyvec.Vector) {
	return e.Encode(samples...)
}
