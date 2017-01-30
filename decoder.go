package tweetenc

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
)

// A Decoder decodes vectors into strings of bytes.
type Decoder struct {
	// Block is used to decode the input vector.
	//
	// The only stateful sub-blocks should be LSTMs.
	Block anyrnn.Stack

	// StateMapper maps the vectors from the encoder to state
	// vectors for the decoder block.
	StateMapper anynet.Layer
}

// NewDecoder creates a Decoder with a default structure.
func NewDecoder(c anyvec.Creator, encodedSize, stateSize int) *Decoder {
	scaler := c.MakeNumeric(16)
	return &Decoder{
		Block: anyrnn.Stack{
			anyrnn.NewLSTM(c, 0x100, stateSize).ScaleInWeights(scaler),
			anyrnn.NewLSTM(c, stateSize, stateSize).ScaleInWeights(scaler),
			anyrnn.NewLSTM(c, stateSize, stateSize).ScaleInWeights(scaler),
			&anyrnn.LayerBlock{
				Layer: anynet.Net{
					anynet.NewFC(c, stateSize, 0x100),
					anynet.LogSoftmax,
				},
			},
		},
		StateMapper: anynet.Net{
			anynet.NewFC(c, encodedSize, stateSize*6),
		},
	}
}

// Guided decodes the batch of vectors and produces
// sequences in a guided fashion.
// It is meant to be used during training, when the
// correct outputs are already known.
//
// The guide sequence should be the actual inputs that
// the decoder would receive if it were producing the
// exact correct result at every timestep.
func (d *Decoder) Guided(encoded anydiff.Res, guide anyseq.Seq, batchSize int) anyseq.Seq {
	mapped := d.StateMapper.Apply(encoded, batchSize)
	start := d.vecToState(mapped.Output(), batchSize)
	startProp := func(sg anyrnn.StateGrad, g anydiff.Grad) {
		ds := d.stateToVec(sg, batchSize)
		mapped.Propagate(ds, g)
	}
	return anyrnn.MapWithStart(guide, d.Block, start, startProp)
}

// Unguided reconstructs a sequence from a feature vector
// without any guiding input sequence.
func (d *Decoder) Unguided(encoded anyvec.Vector) []byte {
	mapped := d.StateMapper.Apply(anydiff.NewConst(encoded), 1)
	state := d.vecToState(mapped.Output(), 1)
	input := oneHot(encoded.Creator(), 0)
	res := []byte{}
	for {
		next := d.Block.Step(state, input)
		state = next.State()

		max := byte(anyvec.MaxIndex(next.Output()))
		if max == 0 {
			break
		}
		res = append(res, max)
		input = oneHot(encoded.Creator(), max)
	}
	return res
}

func (d *Decoder) vecToState(vec anyvec.Vector, batchSize int) anyrnn.State {
	cols := vec.Len() / batchSize
	perBatch := make([]anyvec.Vector, batchSize)
	for i := 0; i < batchSize; i++ {
		perBatch[i] = vec.Slice(cols*i, cols*(i+1))
	}

	outState := make(anyrnn.StackState, len(d.Block))
	for i, layer := range d.Block {
		lstm, ok := layer.(*anyrnn.LSTM)
		if !ok {
			outState[i] = layer.Start(batchSize)
			continue
		}
		start := lstm.Start(batchSize).(*anyrnn.LSTMState)
		moveIntoVecState(perBatch, start.Internal)
		moveIntoVecState(perBatch, start.LastOut)
		outState[i] = start
	}

	return outState
}

func (d *Decoder) stateToVec(s anyrnn.StateGrad, batchSize int) anyvec.Vector {
	perBatch := make([]anyvec.Vector, batchSize)

	for _, subState := range s.(anyrnn.StackGrad) {
		lstmState, ok := subState.(*anyrnn.LSTMState)
		if !ok {
			continue
		}
		moveOutOfVecState(perBatch, lstmState.Internal)
		moveOutOfVecState(perBatch, lstmState.LastOut)
	}

	return perBatch[0].Creator().Concat(perBatch...)
}

func moveIntoVecState(packed []anyvec.Vector, v *anyrnn.VecState) {
	sliceAmount := v.Vector.Len() / len(packed)
	var joinMe []anyvec.Vector
	for i, x := range packed {
		joinMe = append(joinMe, x.Slice(0, sliceAmount))
		packed[i] = x.Slice(sliceAmount, x.Len())
	}
	v.Vector = v.Vector.Creator().Concat(joinMe...)
}

func moveOutOfVecState(packed []anyvec.Vector, v *anyrnn.VecState) {
	sliceAmount := v.Vector.Len() / len(packed)
	for i, x := range packed {
		part := v.Vector.Slice(sliceAmount*i, sliceAmount*(i+1))
		if x == nil {
			packed[i] = part
		} else {
			packed[i] = x.Creator().Concat(x, part)
		}
	}
}
