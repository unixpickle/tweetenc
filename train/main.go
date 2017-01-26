package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/tweetenc"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	var dataPath string
	var encPath string
	var decPath string
	var latent int
	var batchSize int
	var stepSize float64

	flag.StringVar(&dataPath, "data", "", "data CSV file")
	flag.StringVar(&encPath, "encoder", "enc_out", "encoder network path")
	flag.StringVar(&decPath, "decoder", "dec_out", "decoder network path")
	flag.IntVar(&latent, "latent", 128, "latent vector size")
	flag.IntVar(&batchSize, "batch", 16, "SGD batch size")
	flag.Float64Var(&stepSize, "step", 0.001, "SGD step size")

	flag.Parse()

	if dataPath == "" {
		fmt.Fprintln(os.Stderr, "Missing -data flag. See -help for more.")
		os.Exit(1)
	}

	enc, dec := createOrLoad(encPath, decPath, latent)

	log.Println("Loading samples...")
	samples, err := tweetenc.ReadSampleList(dataPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	log.Println("Loaded", samples.Len(), "samples")

	tr := &tweetenc.Trainer{
		Encoder: enc,
		Decoder: dec,
	}

	var iter int
	s := anysgd.SGD{
		Fetcher:     tr,
		Gradienter:  tr,
		Transformer: &anysgd.Adam{},
		Samples:     samples,
		Rater:       anysgd.ConstRater(stepSize),
		BatchSize:   batchSize,
		StatusFunc: func(b anysgd.Batch) {
			log.Printf("iter %d: cost=%v", iter, tr.LastCost)
			iter++
		},
	}

	log.Println("Press Ctrl+C to stop.")
	s.Run(rip.NewRIP().Chan())

	log.Println("Saving...")

	if err := serializer.SaveAny(encPath, enc.Block); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to save encoder:", err)
		os.Exit(1)
	}
	if err := serializer.SaveAny(decPath, dec.Block, dec.StateMapper); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to save decoder:", err)
		os.Exit(1)
	}
}

func createOrLoad(enc, dec string, latent int) (*tweetenc.Encoder, *tweetenc.Decoder) {
	encRes := &tweetenc.Encoder{}
	if err := serializer.LoadAny(enc, &encRes.Block); err != nil {
		log.Println("Creating new encoder...")
		encRes = tweetenc.NewEncoder(anyvec32.CurrentCreator(), latent)
	}

	decRes := &tweetenc.Decoder{}
	if err := serializer.LoadAny(dec, &decRes.Block, &decRes.StateMapper); err != nil {
		log.Println("Creating new decoder...")
		decRes = tweetenc.NewDecoder(anyvec32.CurrentCreator(), latent)
	}

	return encRes, decRes
}
