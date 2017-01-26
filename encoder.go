package tweetenc

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
)

// An Encoder encodes strings of bytes into vectors.
type Encoder struct {
	Block anyrnn.Block
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
