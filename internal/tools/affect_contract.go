package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/Sun668/AMAC/internal/affectcontract"
	"github.com/Sun668/AMAC/internal/approval"
	"github.com/Sun668/AMAC/internal/schema"
)

type AffectContractTool struct {
	mu       sync.Mutex
	sessions map[string]*affectcontract.Contract
}

type affectContractArgs struct {
	Operation              string  `json:"operation"`
	SessionID              string  `json:"session_id"`
	Threshold              float64 `json:"threshold"`
	Margin                 float64 `json:"margin"`
	Modality               string  `json:"modality"`
	ProvisionalState       string  `json:"provisional_state"`
	CorrectnessProbability float64 `json:"correctness_probability"`
}

type affectContractResponse struct {
	Code     int                      `json:"code"`
	Message  string                   `json:"message"`
	Decision *affectcontract.Decision `json:"decision,omitempty"`
}

func NewAffectContractTool() *AffectContractTool {
	return &AffectContractTool{sessions: make(map[string]*affectcontract.Contract)}
}

func (t *AffectContractTool) Name() string { return "affect_contract" }

func (t *AffectContractTool) Definition() schema.ToolDefinition {
	return schema.ToolDefinition{
		Name:        t.Name(),
		Description: "执行异步多模态情感提交契约。先 start 冻结阈值，再按到达顺序 observe；第三模态强制提交完整状态。",
		InputSchema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"operation":               map[string]interface{}{"type": "string", "enum": []string{"start", "observe", "reset"}},
				"session_id":              map[string]interface{}{"type": "string"},
				"threshold":               map[string]interface{}{"type": "number", "minimum": 0, "maximum": 1},
				"margin":                  map[string]interface{}{"type": "number", "minimum": 0, "maximum": 1},
				"modality":                map[string]interface{}{"type": "string", "enum": []string{"T", "A", "V"}},
				"provisional_state":       map[string]interface{}{"type": "string"},
				"correctness_probability": map[string]interface{}{"type": "number", "minimum": 0, "maximum": 1},
			},
			"required": []string{"operation", "session_id"},
		},
	}
}

func (t *AffectContractTool) Execute(_ context.Context, raw json.RawMessage) (string, error) {
	var input affectContractArgs
	if err := json.Unmarshal(raw, &input); err != nil {
		return "", fmt.Errorf("情感契约参数解析失败: %w", err)
	}
	if input.SessionID == "" {
		return "", fmt.Errorf("情感契约会话标识不能为空")
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	switch input.Operation {
	case "start":
		contract, err := affectcontract.New(input.Threshold, input.Margin)
		if err != nil {
			return "", err
		}
		t.sessions[input.SessionID] = contract
		return marshalAffectResponse(affectContractResponse{Code: 0, Message: "情感契约会话已建立"})
	case "observe":
		contract, ok := t.sessions[input.SessionID]
		if !ok {
			return "", fmt.Errorf("情感契约会话不存在，请先执行 start")
		}
		decision, err := contract.Observe(affectcontract.Observation{Modality: input.Modality, ProvisionalState: input.ProvisionalState, CorrectnessProbability: input.CorrectnessProbability})
		if err != nil {
			return "", err
		}
		return marshalAffectResponse(affectContractResponse{Code: 0, Message: "情感契约决策完成", Decision: &decision})
	case "reset":
		delete(t.sessions, input.SessionID)
		return marshalAffectResponse(affectContractResponse{Code: 0, Message: "情感契约会话已重置"})
	default:
		return "", fmt.Errorf("情感契约操作必须是 start、observe 或 reset")
	}
}

func marshalAffectResponse(response affectContractResponse) (string, error) {
	encoded, err := json.Marshal(response)
	if err != nil {
		return "", fmt.Errorf("情感契约结果序列化失败: %w", err)
	}
	return string(encoded), nil
}

func (t *AffectContractTool) RiskLevel() approval.RiskLevel { return approval.RiskSafe }
