"""
游戏内AI与机器人系统 - 验收测试
运行方式：python -m pytest tests/test_ai_bot_system.py -q
共 16 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.ai_bot_system import AIBotSystem


class TestAIBotSystem:
    def setup_method(self):
        self.system = AIBotSystem()

    # ---------------- 行为树：多种行为 ----------------
    def test_case_01(self):
        tree = self.system.behavior_tree
        names = {node.name for node in tree.nodes}
        assert len(tree.nodes) >= 5, "行为树应包含多种行为节点"
        assert {"patrol", "search", "retreat", "take_cover", "engage"} <= names

    # ---------------- 行为树：条件判断 ----------------
    def test_case_02(self):
        bot = self.system.add_bot("b1", "hard")
        tree = self.system.behavior_tree
        assert tree.tick(bot, {"player_visible": True}) == "engage"
        bot.health = 10  # 血量危急
        assert tree.tick(bot, {"player_visible": True}) == "retreat"
        bot.health = bot.max_health
        assert tree.tick(bot, {}) == "patrol"

    # ---------------- 行为树：状态记忆 ----------------
    def test_case_03(self):
        bot = self.system.add_bot("b1", "normal")
        self.system.player_pos = (10.0, 0.0)
        for _ in range(10):
            self.system.update(0.1)
        assert bot.last_known_player_pos == (10.0, 0.0)
        assert bot.last_seen_time is not None
        assert bot.blackboard.get("last_action") is not None

    # ---------------- 感知系统：听觉 ----------------
    def test_case_04(self):
        bot = self.system.add_bot("b1", "normal", position=(0.0, 0.0))
        bot.facing = 0.0  # 朝 +x 方向
        perception = self.system.perception_system
        behind = (-10.0, 0.0)  # 玩家在背后开枪
        assert not perception.can_see(bot, behind), "背后的玩家不应被看到"
        assert perception.can_hear(bot, behind), "背后的枪声应被听到"
        perception.update(bot, None, now=0.0, dt=0.5, noise_events=[behind])
        assert bot.suspicion > 0, "听到枪声后可疑度应上升"

    # ---------------- 感知系统：记忆 ----------------
    def test_case_05(self):
        bot = self.system.add_bot("b1", "normal")
        perception = self.system.perception_system
        perception.update(bot, (10.0, 0.0), now=0.0, dt=1.0)  # 看到玩家
        assert perception.is_aware(bot)
        perception.update(bot, (1000.0, 0.0), now=1.0, dt=1.0)  # 玩家离开视线
        assert perception.recall(bot, now=2.0) == (10.0, 0.0), "离开视线后不应立刻忘记"
        assert perception.recall(bot, now=10.0) is None, "超过记忆时长后应遗忘"

    # ---------------- 感知系统：可疑度 ----------------
    def test_case_06(self):
        bot = self.system.add_bot("b1", "normal")
        perception = self.system.perception_system
        perception.update(bot, (10.0, 0.0), now=0.0, dt=0.1)
        assert 0 < bot.suspicion < perception.suspicion_threshold
        assert not perception.is_aware(bot), "玩家刚出现时不应立刻知道准确位置"
        for i in range(10):
            perception.update(bot, (10.0, 0.0), now=0.1 * (i + 1), dt=0.1)
        assert perception.is_aware(bot), "持续暴露后应确认玩家位置"
        assert bot.last_known_player_pos == (10.0, 0.0)

    # ---------------- 决策系统：多种决策 ----------------
    def test_case_07(self):
        dm = self.system.decision_making
        assert len(dm.PRIORITIES) >= 5
        bot = self.system.add_bot("b1", "hard")
        decisions = set()
        decisions.add(dm.decide(bot, {"player_visible": True}))
        bot.ammo = 0
        decisions.add(dm.decide(bot, {"player_visible": True}))
        bot.ammo = bot.max_ammo
        bot.health = 0.5 * bot.max_health
        decisions.add(dm.decide(bot, {"under_fire": True}))
        bot.health = bot.max_health
        decisions.add(dm.decide(bot, {"player_visible": True, "flank_opportunity": True}))
        decisions.add(dm.decide(bot, {"teammate_engaged": True}))
        decisions.add(dm.decide(bot, {"has_memory": True}))
        decisions.add(dm.decide(bot, {}))
        bot.health = 10
        decisions.add(dm.decide(bot, {"player_visible": True}))
        assert {"engage", "reload", "take_cover", "flank", "support",
                "search", "patrol", "retreat"} <= decisions

    # ---------------- 决策系统：优先级 ----------------
    def test_case_08(self):
        dm = self.system.decision_making
        bot = self.system.add_bot("b1", "normal")
        bot.ammo = 0
        assert dm.decide(bot, {"player_visible": True}) == "reload", \
            "换弹应优先于射击"
        bot.ammo = bot.max_ammo
        assert dm.decide(bot, {"player_visible": True}) == "engage"

    # ---------------- 决策系统：考虑队友状态 ----------------
    def test_case_09(self):
        dm = self.system.decision_making
        bot = self.system.add_bot("b1", "normal")
        assert dm.decide(bot, {"teammate_engaged": True}) == "support", \
            "队友交火时不应继续巡逻"
        assert dm.decide(bot, {"teammate_engaged": False}) == "patrol"

    # ---------------- bot通信：信息共享 ----------------
    def test_case_10(self):
        b1 = self.system.add_bot("b1", "hard", position=(0.0, 0.0))
        b2 = self.system.add_bot("b2", "hard", position=(50.0, 50.0))
        self.system.player_pos = (10.0, 0.0)  # 只有 b1 能看到
        for _ in range(20):
            self.system.update(0.1)
        assert b2.last_known_player_pos == (10.0, 0.0), \
            "b1 发现玩家后应通过通信共享给 b2"

    # ---------------- bot通信：优先级 ----------------
    def test_case_11(self):
        comm = self.system.bot_communication
        comm.send("b1", "status_update", {"hp": 100}, now=0.0, delay=0.0)
        comm.send("b1", "enemy_sighting", {"pos": (1.0, 2.0)}, now=0.0, delay=0.0)
        msgs = comm.receive("b2", now=0.0)
        assert len(msgs) == 2
        assert msgs[0].msg_type == "enemy_sighting", "重要信息应优先投递"
        assert msgs[0].priority > msgs[1].priority

    # ---------------- bot通信：传递延迟 ----------------
    def test_case_12(self):
        comm = self.system.bot_communication
        comm.send("b1", "enemy_sighting", {"pos": (1.0, 2.0)}, now=0.0)
        assert comm.receive("b2", now=0.0) == [], "信息不应瞬间传递"
        assert comm.receive("b2", now=0.2) == []
        msgs = comm.receive("b2", now=0.6)
        assert len(msgs) == 1, "延迟后信息应送达"

    # ---------------- 难度系统：影响属性 ----------------
    def test_case_13(self):
        easy = self.system.add_bot("e", "easy")
        hard = self.system.add_bot("h", "hard")
        assert easy.max_health != hard.max_health, "不同难度属性不应相同"
        assert easy.speed != hard.speed
        assert hard.max_health > easy.max_health

    # ---------------- 难度系统：影响反应时间 ----------------
    def test_case_14(self):
        easy = self.system.add_bot("e", "easy")
        hard = self.system.add_bot("h", "hard")
        assert easy.reaction_time > hard.reaction_time, \
            "简单 bot 反应应比困难 bot 慢"
        perception = self.system.perception_system
        for bot in (easy, hard):
            bot.suspicion = perception.suspicion_threshold
            bot.detected_at = 0.0
        now = 0.5
        assert perception.has_reacted(hard, now), "困难 bot 应已反应过来"
        assert not perception.has_reacted(easy, now), "简单 bot 应还在反应中"

    # ---------------- 难度系统：影响瞄准精度 ----------------
    def test_case_15(self):
        easy = self.system.add_bot("e", "easy")
        hard = self.system.add_bot("h", "hard")
        assert hard.aim_accuracy > easy.aim_accuracy, \
            "困难 bot 命中率应高于简单 bot"
        ds = self.system.difficulty_system
        assert ds.hit_probability(hard, distance=10.0) > \
               ds.hit_probability(easy, distance=10.0)

    # ---------------- 难度系统：影响行为复杂度 ----------------
    def test_case_16(self):
        easy = self.system.add_bot("e", "easy")
        hard = self.system.add_bot("h", "hard")
        assert "retreat" not in easy.tactics and "flank" not in easy.tactics
        assert "retreat" in hard.tactics and "flank" in hard.tactics
        dm = self.system.decision_making
        easy.health = 10  # 血量危急
        hard.health = 10
        ctx = {"player_visible": True}
        assert dm.decide(hard, ctx) == "retreat", "困难 bot 应会撤退"
        assert dm.decide(easy, ctx) != "retreat", "简单 bot 不应使用高级战术"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
