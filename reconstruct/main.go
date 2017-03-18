// Command reconstruct takes a tweet's contents, encodes
// it and decodes it again, and displays the result.
//
// The command also has the ability to interpolate between
// two tweets in latent feature space.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/unixpickle/serializer"
	"github.com/unixpickle/tweetenc"
)

func main() {
	var encFile string
	var decFile string
	var startStr string

	var numStops int
	var endStr string

	flag.StringVar(&encFile, "encoder", "../train/enc_out", "encoder input file")
	flag.StringVar(&decFile, "decoder", "../train/dec_out", "decoder input file")
	flag.StringVar(&startStr, "tweet", "", "tweet body")
	flag.IntVar(&numStops, "stops", 1, "interpolation stops")
	flag.StringVar(&endStr, "end", "", "end tweet body for interpolation")

	flag.Parse()

	if numStops < 1 {
		flag.PrintDefaults()
		os.Exit(1)
	}

	if startStr == "" {
		fmt.Fprintln(os.Stderr, "Missing -tweet flag. See -help for more.")
		os.Exit(1)
	}

	enc := &tweetenc.Encoder{}
	if err := serializer.LoadAny(encFile, &enc); err != nil {
		fmt.Fprintln(os.Stderr, "load encoder:", err)
		os.Exit(1)
	}

	dec := &tweetenc.Decoder{}
	if err := serializer.LoadAny(decFile, &dec); err != nil {
		fmt.Fprintln(os.Stderr, "load decoder:", err)
		os.Exit(1)
	}

	if numStops != 1 {
		interpolate(startStr, endStr, enc, dec, numStops)
	} else {
		encoded, _ := enc.Encode(startStr)
		decoded := dec.Unguided(encoded)
		fmt.Println("Decoded to:", string(decoded))
	}
}

func interpolate(start, end string, enc *tweetenc.Encoder, dec *tweetenc.Decoder, stops int) {
	startVec, _ := enc.Encode(start)
	endVec, _ := enc.Encode(end)

	for i := 0; i < stops; i++ {
		fracDone := float64(i) / float64(stops-1)
		vec1 := startVec.Copy()
		vec1.Scale(vec1.Creator().MakeNumeric(1 - fracDone))
		vec2 := endVec.Copy()
		vec2.Scale(vec2.Creator().MakeNumeric(fracDone))
		vec1.Add(vec2)

		fmt.Printf("%.3f: %s\n", fracDone, string(dec.Unguided(vec1)))
	}
}
