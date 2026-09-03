package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/Sun668/AMAC/internal/affectcontract"
	projecttools "github.com/Sun668/AMAC/internal/tools"
)

type observation struct {
	Modality               string  `json:"modality"`
	ProvisionalState       string  `json:"provisional_state"`
	CorrectnessProbability float64 `json:"correctness_probability"`
}

type replayRecord struct {
	SessionID       string        `json:"session_id"`
	ClipID          string        `json:"clip_id"`
	Path            string        `json:"path"`
	Threshold       float64       `json:"threshold"`
	Margin          float64       `json:"margin"`
	Observations    []observation `json:"observations"`
	ExpectedCommits []*string     `json:"expected_commits"`
}

type toolResponse struct {
	Code     int                      `json:"code"`
	Message  string                   `json:"message"`
	Decision *affectcontract.Decision `json:"decision"`
}

type mismatch struct {
	SessionID string  `json:"session_id"`
	Stage     int     `json:"stage"`
	Expected  *string `json:"expected"`
	Observed  *string `json:"observed"`
	Error     string  `json:"error,omitempty"`
}

func execute(tool *projecttools.AffectContractTool, payload map[string]interface{}) (toolResponse, error) {
	raw, err := json.Marshal(payload)
	if err != nil {
		return toolResponse{}, fmt.Errorf("请求序列化失败: %w", err)
	}
	value, err := tool.Execute(context.Background(), raw)
	if err != nil {
		return toolResponse{}, err
	}
	var response toolResponse
	if err := json.Unmarshal([]byte(value), &response); err != nil {
		return toolResponse{}, fmt.Errorf("工具响应解析失败: %w", err)
	}
	return response, nil
}

func same(left, right *string) bool {
	if left == nil || right == nil {
		return left == nil && right == nil
	}
	return *left == *right
}

func expectError(tool *projecttools.AffectContractTool, payload map[string]interface{}) bool {
	_, err := execute(tool, payload)
	return err != nil
}

func robustnessChecks() map[string]bool {
	tool := projecttools.NewAffectContractTool()
	checks := map[string]bool{}
	checks["observe_before_start_rejected"] = expectError(tool, map[string]interface{}{"operation": "observe", "session_id": "missing", "modality": "T", "provisional_state": "positive", "correctness_probability": 0.9})
	checks["invalid_threshold_rejected"] = expectError(tool, map[string]interface{}{"operation": "start", "session_id": "bad-threshold", "threshold": 1.1, "margin": 0.0})
	_, startErr := execute(tool, map[string]interface{}{"operation": "start", "session_id": "robust", "threshold": 0.5, "margin": 0.1})
	_, firstErr := execute(tool, map[string]interface{}{"operation": "observe", "session_id": "robust", "modality": "T", "provisional_state": "positive", "correctness_probability": 0.9})
	checks["valid_start_and_observe"] = startErr == nil && firstErr == nil
	checks["duplicate_modality_rejected"] = expectError(tool, map[string]interface{}{"operation": "observe", "session_id": "robust", "modality": "T", "provisional_state": "positive", "correctness_probability": 0.9})
	checks["invalid_probability_rejected"] = expectError(tool, map[string]interface{}{"operation": "observe", "session_id": "robust", "modality": "A", "provisional_state": "positive", "correctness_probability": -0.1})
	checks["invalid_modality_rejected"] = expectError(tool, map[string]interface{}{"operation": "observe", "session_id": "robust", "modality": "X", "provisional_state": "positive", "correctness_probability": 0.9})
	_, secondErr := execute(tool, map[string]interface{}{"operation": "observe", "session_id": "robust", "modality": "A", "provisional_state": "positive", "correctness_probability": 0.9})
	final, finalErr := execute(tool, map[string]interface{}{"operation": "observe", "session_id": "robust", "modality": "V", "provisional_state": "negative", "correctness_probability": 0.0})
	checks["third_modality_forces_final_revision"] = secondErr == nil && finalErr == nil && final.Decision != nil && final.Decision.Final && final.Decision.Action == affectcontract.ActionRevise && final.Decision.CommittedState != nil && *final.Decision.CommittedState == "negative"
	_, resetErr := execute(tool, map[string]interface{}{"operation": "reset", "session_id": "robust"})
	checks["reset_removes_session"] = resetErr == nil && expectError(tool, map[string]interface{}{"operation": "observe", "session_id": "robust", "modality": "T", "provisional_state": "positive", "correctness_probability": 0.9})
	return checks
}

func fileSHA256(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}

func percentile(values []float64, fraction float64) float64 {
	if len(values) == 0 {
		return 0
	}
	position := int(float64(len(values)-1)*fraction + 0.5)
	return values[position]
}

func main() {
	input := flag.String("input", "", "回放 JSONL 输入")
	output := flag.String("output", "", "结果 JSON 输出")
	flag.Parse()
	if *input == "" || *output == "" {
		fmt.Fprintln(os.Stderr, "必须提供 --input 和 --output")
		os.Exit(2)
	}
	if _, err := os.Stat(*output); err == nil {
		fmt.Fprintln(os.Stderr, "结果文件已存在，禁止覆盖")
		os.Exit(2)
	}
	started := time.Now()
	handle, err := os.Open(*input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "回放输入打开失败: %v\n", err)
		os.Exit(1)
	}
	defer handle.Close()
	tool := projecttools.NewAffectContractTool()
	scanner := bufio.NewScanner(handle)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	latencies := make([]float64, 0, 20000)
	mismatches := make([]mismatch, 0)
	mismatchCount := 0
	paths := 0
	observations := 0
	finalIdentity := true
	for scanner.Scan() {
		var record replayRecord
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			fmt.Fprintf(os.Stderr, "回放记录解析失败: %v\n", err)
			os.Exit(1)
		}
		if len(record.Observations) != 3 || len(record.ExpectedCommits) != 3 {
			fmt.Fprintln(os.Stderr, "回放记录必须包含三个观察和三个期望状态")
			os.Exit(1)
		}
		if _, err := execute(tool, map[string]interface{}{"operation": "start", "session_id": record.SessionID, "threshold": record.Threshold, "margin": record.Margin}); err != nil {
			fmt.Fprintf(os.Stderr, "Agent 会话建立失败: %v\n", err)
			os.Exit(1)
		}
		for stage, item := range record.Observations {
			callStarted := time.Now()
			response, callErr := execute(tool, map[string]interface{}{"operation": "observe", "session_id": record.SessionID, "modality": item.Modality, "provisional_state": item.ProvisionalState, "correctness_probability": item.CorrectnessProbability})
			latencies = append(latencies, float64(time.Since(callStarted).Nanoseconds())/1000.0)
			observations++
			var observed *string
			if callErr == nil && response.Decision != nil {
				observed = response.Decision.CommittedState
				if stage == 2 && (!response.Decision.Final || observed == nil || *observed != item.ProvisionalState) {
					finalIdentity = false
				}
			}
			if callErr != nil || !same(record.ExpectedCommits[stage], observed) {
				mismatchCount++
				entry := mismatch{SessionID: record.SessionID, Stage: stage + 1, Expected: record.ExpectedCommits[stage], Observed: observed}
				if callErr != nil {
					entry.Error = callErr.Error()
				}
				if len(mismatches) < 50 {
					mismatches = append(mismatches, entry)
				}
			}
		}
		_, _ = execute(tool, map[string]interface{}{"operation": "reset", "session_id": record.SessionID})
		paths++
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "回放输入读取失败: %v\n", err)
		os.Exit(1)
	}
	sort.Float64s(latencies)
	checks := robustnessChecks()
	allRobust := true
	for _, passed := range checks {
		allRobust = allRobust && passed
	}
	inputHash, err := fileSHA256(*input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "输入哈希计算失败: %v\n", err)
		os.Exit(1)
	}
	result := map[string]interface{}{
		"schema":                    "affect-contract-agent-replay-v1",
		"code":                      0,
		"message":                   "Agent 情感契约回放完成",
		"input_sha256":              inputHash,
		"paths":                     paths,
		"observations":              observations,
		"trajectory_mismatch_count": mismatchCount,
		"mismatch_examples":         mismatches,
		"final_identity":            finalIdentity,
		"robustness_checks":         checks,
		"robustness_all_passed":     allRobust,
		"latency_microseconds":      map[string]float64{"p50": percentile(latencies, 0.50), "p95": percentile(latencies, 0.95), "p99": percentile(latencies, 0.99), "max": percentile(latencies, 1.0)},
		"latency_scope":             "当前机器进程内工具调用描述值，不代表网络或端到端延迟",
		"elapsed_seconds":           time.Since(started).Seconds(),
	}
	encoded, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "结果序列化失败: %v\n", err)
		os.Exit(1)
	}
	if err := os.MkdirAll(filepath.Dir(*output), 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "结果目录创建失败: %v\n", err)
		os.Exit(1)
	}
	if err := os.WriteFile(*output, append(encoded, '\n'), 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "结果写入失败: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(string(encoded))
}
