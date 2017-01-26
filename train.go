package tweetenc

import (
	"crypto/md5"
	"errors"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
)

// A SampleList is a list of textual samples.
type SampleList [][]byte

// Len returns the sample count.
func (s SampleList) Len() int {
	return len(s)
}

// Swap swaps two samples.
func (s SampleList) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// Slice returns a shallow copy of the slice.
func (s SampleList) Slice(start, end int) anysgd.SampleList {
	return append(SampleList{}, s[start:end]...)
}

// Hash returns a hash of the sample.
func (s SampleList) Hash(i int) []byte {
	res := md5.Sum(s[i])
	return res[:]
}

// A Trainer is an anysgd.Fetcher and anysgd.Gradienter
// for training an encoder/decoder pair.
type Trainer struct {
	Encoder *Encoder
	Decoder *Decoder

	LastCost anyvec.Numeric
}

// Fetch produces a batch that represents the training
// samples in the SampleList.
func (t *Trainer) Fetch(s anysgd.SampleList) (anysgd.Batch, error) {
	cr := t.creator()
	zero := oneHot(cr, 0)

	inSeqs := make([][]anyvec.Vector, s.Len())
	guideSeqs := make([][]anyvec.Vector, s.Len())
	for i := range inSeqs {
		data := s.(SampleList)[i]
		if len(data) == 0 {
			return nil, errors.New("encountered empty sample string")
		}
		seq := []anyvec.Vector{zero}
		for _, x := range data {
			seq = append(seq, oneHot(cr, x))
		}
		seq = append(seq, zero)
		inSeqs[i] = seq[1:]
		guideSeqs[i] = seq[:len(seq)-1]
	}

	return &trainerBatch{
		InSeqs: anyseq.ConstSeqList(inSeqs),
		Guide:  anyseq.ConstSeqList(guideSeqs),
	}, nil
}

// TotalCost computes the average cost for the batch.
func (t *Trainer) TotalCost(b anysgd.Batch) anydiff.Res {
	tb := b.(*trainerBatch)
	batchSize := len(tb.InSeqs.Output()[0].Present)
	encoded := t.Encoder.Apply(tb.InSeqs)
	decoded := t.Decoder.Guided(encoded, tb.Guide, batchSize)

	var idx int
	var costCount int
	allCosts := anyseq.Map(decoded, func(a anydiff.Res, n int) anydiff.Res {
		desired := tb.InSeqs.Output()[idx]
		costCount += desired.NumPresent()
		idx++
		c := anynet.DotCost{}.Cost(anydiff.NewConst(desired.Packed), a, n)
		return c
	})

	sum := anydiff.Sum(anyseq.Sum(allCosts))
	scaler := sum.Output().Creator().MakeNumeric(1 / float64(costCount))
	return anydiff.Scale(sum, scaler)
}

// Gradient computes a gradient for the batch and also
// sets t.LastCost.
func (t *Trainer) Gradient(b anysgd.Batch) anydiff.Grad {
	res := anydiff.Grad{}
	parts := []interface{}{t.Encoder.Block, t.Decoder.Block, t.Decoder.StateMapper}
	for _, part := range parts {
		if parameterizer, ok := part.(anynet.Parameterizer); ok {
			for _, p := range parameterizer.Parameters() {
				res[p] = p.Vector.Creator().MakeVector(p.Vector.Len())
			}
		}
	}
	cost := t.TotalCost(b)
	data := cost.Output().Creator().MakeNumericList([]float64{1})
	upstream := cost.Output().Creator().MakeVectorData(data)
	cost.Propagate(upstream, res)
	return res
}

func (t *Trainer) creator() anyvec.Creator {
	return t.Decoder.Block.Parameters()[0].Vector.Creator()
}

type trainerBatch struct {
	InSeqs anyseq.Seq
	Guide  anyseq.Seq
}

func oneHot(c anyvec.Creator, b byte) anyvec.Vector {
	data := make([]float64, 0x100)
	data[int(b)] = 1
	return c.MakeVectorData(c.MakeNumericList(data))
}