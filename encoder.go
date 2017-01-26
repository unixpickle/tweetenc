package tweetenc

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
)

// An Encoder encodes strings of bytes into vectors.
type Encoder struct {
	Block anyrnn.Block
}

// NewEncoder creates an Encoder.
func NewEncoder(c anyvec.Creator, encodedSize int) *Encoder {
	return &Encoder{
		Block: anyrnn.Stack{
			anyrnn.NewLSTM(c, 0x100, 512),
			anyrnn.NewLSTM(c, 512, 512),
			&anyrnn.LayerBlock{
				Layer: anynet.Net{
					anynet.NewFC(c, 512, encodedSize),
					anynet.Tanh,
				},
			},
		},
	}
}

// Apply applies the encoder to an input sequence.
//
// There must be at least one sequence, and all sequences
// must be non-empty.
func (e *Encoder) Apply(seqs anyseq.Seq) anydiff.Res {
	if len(seqs.Output()) == 0 {
		panic("must have at least one sequence")
	}
	if seqs.Output()[0].NumPresent() != len(seqs.Output()[0].Present) {
		panic("input sequences must be non-empty")
	}
	outSeq := anyrnn.Map(seqs, e.Block)
	return anyseq.Tail(outSeq)
}
