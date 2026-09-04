package affectcontract

import "testing"

func TestObserveDistinguishesWaitAndHold(t *testing.T) {
	contract, err := New(0.7, 0.1)
	if err != nil {
		t.Fatalf("创建情感契约失败: %v", err)
	}

	wait, err := contract.Observe(Observation{Modality: "T", ProvisionalState: "neutral", CorrectnessProbability: 0.6})
	if err != nil {
		t.Fatalf("首次观察失败: %v", err)
	}
	if wait.Action != ActionWait || wait.CommittedState != nil {
		t.Fatalf("未提交状态应返回 WAIT，实际为 %s", wait.Action)
	}

	commit, err := contract.Observe(Observation{Modality: "A", ProvisionalState: "positive", CorrectnessProbability: 0.8})
	if err != nil {
		t.Fatalf("第二次观察失败: %v", err)
	}
	if commit.Action != ActionCommit {
		t.Fatalf("首次提交应返回 COMMIT，实际为 %s", commit.Action)
	}

	hold, err := contract.Observe(Observation{Modality: "V", ProvisionalState: "positive", CorrectnessProbability: 0.4})
	if err != nil {
		t.Fatalf("终端观察失败: %v", err)
	}
	if hold.Action != ActionHold || hold.CommittedState == nil || *hold.CommittedState != "positive" {
		t.Fatalf("未改变的已提交状态应返回 HOLD，实际为 %s", hold.Action)
	}
}

func TestObserveReturnsHoldBeforeTerminalRevision(t *testing.T) {
	contract, err := New(0.7, 0.2)
	if err != nil {
		t.Fatalf("创建情感契约失败: %v", err)
	}

	if _, err := contract.Observe(Observation{Modality: "T", ProvisionalState: "positive", CorrectnessProbability: 0.8}); err != nil {
		t.Fatalf("首次观察失败: %v", err)
	}
	hold, err := contract.Observe(Observation{Modality: "A", ProvisionalState: "negative", CorrectnessProbability: 0.8})
	if err != nil {
		t.Fatalf("第二次观察失败: %v", err)
	}
	if hold.Action != ActionHold || hold.CommittedState == nil || *hold.CommittedState != "positive" {
		t.Fatalf("未达到修订门槛时应返回 HOLD，实际为 %s", hold.Action)
	}
}
