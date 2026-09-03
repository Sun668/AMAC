package affectcontract

import (
	"fmt"
	"math"
)

type Action string

const (
	ActionWait   Action = "WAIT"
	ActionCommit Action = "COMMIT"
	ActionHold   Action = "HOLD"
	ActionRevise Action = "REVISE"
)

type Observation struct {
	Modality               string  `json:"modality"`
	ProvisionalState       string  `json:"provisional_state"`
	CorrectnessProbability float64 `json:"correctness_probability"`
}

type Decision struct {
	Action           Action   `json:"action"`
	ProvisionalState string   `json:"provisional_state"`
	CommittedState   *string  `json:"committed_state"`
	SeenModalities   []string `json:"seen_modalities"`
	Final            bool     `json:"final"`
}

type Contract struct {
	threshold float64
	margin    float64
	seen      map[string]struct{}
	order     []string
	committed *string
}

func New(threshold, margin float64) (*Contract, error) {
	if math.IsNaN(threshold) || threshold < 0 || threshold > 1 {
		return nil, fmt.Errorf("提交阈值必须位于 0 到 1 之间")
	}
	if math.IsNaN(margin) || margin < 0 || threshold+margin > 1 {
		return nil, fmt.Errorf("修订边际必须非负，且与提交阈值之和不能超过 1")
	}
	return &Contract{threshold: threshold, margin: margin, seen: make(map[string]struct{})}, nil
}

func (c *Contract) Observe(observation Observation) (Decision, error) {
	if observation.Modality != "T" && observation.Modality != "A" && observation.Modality != "V" {
		return Decision{}, fmt.Errorf("模态必须是 T、A 或 V")
	}
	if _, exists := c.seen[observation.Modality]; exists {
		return Decision{}, fmt.Errorf("同一会话中的模态不能重复到达")
	}
	if observation.ProvisionalState == "" {
		return Decision{}, fmt.Errorf("候选情感状态不能为空")
	}
	if math.IsNaN(observation.CorrectnessProbability) || observation.CorrectnessProbability < 0 || observation.CorrectnessProbability > 1 {
		return Decision{}, fmt.Errorf("当前正确概率必须位于 0 到 1 之间")
	}
	c.seen[observation.Modality] = struct{}{}
	c.order = append(c.order, observation.Modality)
	final := len(c.order) == 3
	action := ActionWait
	if final {
		action = ActionCommit
		if c.committed != nil && *c.committed != observation.ProvisionalState {
			action = ActionRevise
		} else if c.committed != nil {
			action = ActionHold
		}
		c.setCommitted(observation.ProvisionalState)
	} else if c.committed == nil && observation.CorrectnessProbability >= c.threshold {
		action = ActionCommit
		c.setCommitted(observation.ProvisionalState)
	} else if c.committed != nil && *c.committed != observation.ProvisionalState && observation.CorrectnessProbability >= c.threshold+c.margin {
		action = ActionRevise
		c.setCommitted(observation.ProvisionalState)
	} else if c.committed != nil {
		action = ActionHold
	}
	return Decision{Action: action, ProvisionalState: observation.ProvisionalState, CommittedState: clone(c.committed), SeenModalities: append([]string(nil), c.order...), Final: final}, nil
}

func (c *Contract) setCommitted(state string) {
	value := state
	c.committed = &value
}

func clone(value *string) *string {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}
