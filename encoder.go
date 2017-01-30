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
func NewEncoder(c anyvec.Creator, encodedSize, stateSize int) *Encoder {
	scaler := c.MakeNumeric(16)
	return &Encoder{
		Block: anyrnn.Stack{
			anyrnn.NewLSTM(c, 0x100, stateSize).ScaleInWeights(scaler),
			anyrnn.NewLSTM(c, stateSize, stateSize).ScaleInWeights(scaler),
			anyrnn.NewLSTM(c, stateSize, stateSize).ScaleInWeights(scaler),
			&anyrnn.LayerBlock{
				Layer: anynet.Net{
					anynet.NewFC(c, stateSize, encodedSize),
					anynet.Tanh,
				},
			},
		},
	}
}

// Apply applies the encoder to an input sequence, which
// should be reversed and should lack a null-terminator.
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

// Encode encodes strings to a packed vector.
func (e *Encoder) Encode(samples ...string) anyvec.Vector {
	var inSeqs [][]anyvec.Vector
	for _, s := range samples {
		inSeq := []anyvec.Vector{}
		cr := e.Block.(anynet.Parameterizer).Parameters()[0].Vector.Creator()
		byteString := []byte(s)
		for i := len(byteString) - 1; i >= 0; i-- {
			inSeq = append(inSeq, oneHot(cr, byteString[i]))
		}
		inSeqs = append(inSeqs, inSeq)
	}
	seqs := anyseq.ConstSeqList(inSeqs)
	return e.Apply(seqs).Output()
}
