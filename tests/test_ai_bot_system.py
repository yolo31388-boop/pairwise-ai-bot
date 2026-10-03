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

    # ---------- 行为树 ----------

    def test_case_01(self):
        """行为树包含多种行为：巡逻/攻击/搜索/掩护/撤退"""
        # 巡逻：无敌人时
        self.system.add_bot("b1")
        self.system.update(0.1)
        assert self.system.bots["b1"].behavior == "patrol"

        # 攻击：看到玩家且反应完成
        self.system.add_player("p1", (10.0, 0.0))
        self.system.update(1.0)
        assert self.system.bots["b1"].behavior == "attack"

        # 搜索：玩家脱离视线后搜索最后已知位置
        self.system.players["p1"].position = (100.0, 100.0)
        self.system.update(0.1)
        assert self.system.bots["b1"].behavior == "search"

        # 掩护：可见敌人且正在挨打的bot寻找掩护
        self.system.add_bot("b2", (0.0, 5.0))
        self.system.players["p1"].position = (10.0, 0.0)
        self.system.apply_damage("b2", 10.0)
        self.system.update(1.0)
        assert self.system.bots["b2"].behavior == "take_cover"

        # 撤退：困难bot残血时撤退
        self.system.add_bot("b3", (0.0, -5.0), difficulty="hard")
        self.system.apply_damage("b3", 120.0)
        self.system.update(1.0)
        assert self.system.bots["b3"].behavior == "retreat"

    def test_case_02(self):
        """行为树有条件判断：同一bot在不同条件下行为不同"""
        self.system.add_bot("b1", difficulty="hard")
        self.system.add_player("p1", (10.0, 0.0))
        self.system.update(1.0)
        assert self.system.bots["b1"].behavior == "attack"
        # 条件变化（血量过低）后行为切换
        self.system.apply_damage("b1", 120.0)
        self.system.update(0.1)
        assert self.system.bots["b1"].behavior == "retreat"

    def test_case_03(self):
        """行为树有状态记忆：记得玩家最后出现的位置并转为搜索"""
        self.system.add_bot("b1")
        player = self.system.add_player("p1", (10.0, 0.0))
        self.system.update(1.0)
        assert self.system.bots["b1"].behavior == "attack"
        # 玩家消失后，bot 记得最后位置而不是立刻回到巡逻
        player.position = (100.0, 100.0)
        self.system.update(0.1)
        bot = self.system.bots["b1"]
        assert bot.behavior == "search"
        assert bot.blackboard["last_known_position"] == (10.0, 0.0)
        assert "attack" in bot.blackboard["behavior_history"]

    # ---------- 感知系统 ----------

    def test_case_04(self):
        """感知系统有听觉：玩家在bot背后开枪也能被察觉"""
        self.system.add_bot("b1", facing=0.0)  # 面朝 +x 方向
        player = self.system.add_player("p1", (-10.0, 0.0))  # 背后
        assert not self.system.can_see(self.system.bots["b1"], player)
        self.system.emit_sound(player.position, loudness=1.0, source="p1")
        self.system.update(0.5)
        bot = self.system.bots["b1"]
        assert bot.suspicion > 0.0
        assert bot.blackboard["last_known_position"] == (-10.0, 0.0)
        assert bot.behavior == "search"

    def test_case_05(self):
        """感知系统有记忆：玩家离开视线后记忆保留一段时间，超时遗忘"""
        self.system.add_bot("b1")
        player = self.system.add_player("p1", (10.0, 0.0))
        self.system.update(1.0)
        player.position = (100.0, 100.0)  # 脱离视线
        self.system.update(0.1)
        result = self.system.perception_system("b1", 0.1)
        assert result["memory_valid"]
        assert result["last_known_position"] == (10.0, 0.0)
        # 超过记忆时长后遗忘
        self.system.update(6.0)
        result = self.system.perception_system("b1", 0.1)
        assert not result["memory_valid"]
        assert result["last_known_position"] is None

    def test_case_06(self):
        """感知系统有可疑度：逐渐累积，低可疑度时位置估计有误差"""
        self.system.add_bot("b1")
        self.system.add_player("p1", (15.0, 0.0))
        self.system.update(0.1)
        bot = self.system.bots["b1"]
        first = bot.suspicion
        assert 0.0 < first < self.system.ALERT_THRESHOLD
        # 低可疑度：位置估计不精确
        estimate = self.system.estimated_player_position("b1")
        assert estimate is not None and estimate != (15.0, 0.0)
        # 可疑度继续累积
        self.system.update(0.2)
        assert bot.suspicion > first
        # 完全警觉后：获得精确位置
        self.system.update(0.5)
        assert bot.suspicion >= self.system.ALERT_THRESHOLD
        assert self.system.estimated_player_position("b1") == (15.0, 0.0)

    # ---------- 决策系统 ----------

    def test_case_07(self):
        """决策系统有多种决策：巡逻/射击/换弹/掩护/撤退/绕后/搜索"""
        # 巡逻
        self.system.add_bot("b1")
        self.system.update(0.1)
        assert self.system.decision_making("b1") == "patrol"
        # 射击
        self.system.add_player("p1", (10.0, 0.0))
        self.system.update(1.0)
        assert self.system.decision_making("b1") == "shoot"
        # 换弹
        self.system.bots["b1"].ammo = 0
        assert self.system.decision_making("b1") == "reload"
        # 掩护
        self.system.bots["b1"].ammo = 30
        self.system.apply_damage("b1", 10.0)
        assert self.system.decision_making("b1") == "take_cover"
        # 撤退（困难bot残血）
        self.system.add_bot("b2", (0.0, 5.0), difficulty="hard")
        self.system.apply_damage("b2", 120.0)
        self.system.update(0.1)
        assert self.system.decision_making("b2") == "retreat"
        # 绕后（队友已交火）
        self.system.add_bot("b3", (0.0, -5.0), difficulty="hard")
        self.system.update(1.0)
        self.system.update(1.0)  # 等待 b1 受击状态结束，重新回到交火状态
        assert self.system.bots["b1"].behavior == "attack"
        assert self.system.decision_making("b3") == "flank"
        # 搜索（玩家脱离视线）
        self.system.players["p1"].position = (100.0, 100.0)
        self.system.update(0.1)
        assert self.system.decision_making("b1") == "search"

    def test_case_08(self):
        """决策有优先级：换弹优先于射击，撤退优先于换弹"""
        self.system.add_bot("b1")
        self.system.add_player("p1", (10.0, 0.0))
        self.system.update(1.0)
        self.system.bots["b1"].ammo = 0
        # 换弹与射击同时满足时，换弹胜出
        assert self.system.decision_making("b1") == "reload"
        # 残血的困难bot：撤退优先级高于换弹
        self.system.add_bot("b2", (0.0, 5.0), difficulty="hard")
        self.system.bots["b2"].ammo = 0
        self.system.apply_damage("b2", 120.0)
        self.system.update(0.1)
        assert self.system.decision_making("b2") == "retreat"

    def test_case_09(self):
        """决策考虑队友状态：队友交火时bot支援而不是继续巡逻"""
        self.system.add_bot("b1", (0.0, 0.0))
        self.system.add_bot("b2", (0.0, -50.0))  # 远离战场，看不到玩家
        self.system.add_player("p1", (10.0, 0.0))
        self.system.update(1.0)
        assert self.system.bots["b1"].behavior == "attack"
        # b2 看不到玩家，但知道队友在交火，应前往支援
        assert self.system.decision_making("b2") == "support"

    # ---------- bot通信 ----------

    def test_case_10(self):
        """bot通信共享信息：一个bot发现玩家，其他bot获得位置"""
        self.system.add_bot("b1")
        self.system.add_bot("b2", (0.0, -50.0))
        self.system.bot_communication(
            "b1", {"position": (12.0, 8.0)}, msg_type="enemy_spotted", delay=0.1)
        self.system.update(0.2)
        assert self.system.bots["b2"].blackboard["last_known_position"] == (12.0, 8.0)

    def test_case_11(self):
        """通信有优先级：重要消息先于普通消息投递"""
        self.system.add_bot("b1")
        self.system.add_bot("b2")
        self.system.bot_communication("b1", "例行状态", msg_type="status", delay=0.0)
        self.system.bot_communication(
            "b1", {"position": (5.0, 5.0)}, msg_type="enemy_spotted", delay=0.0)
        self.system.update(0.1)
        inbox = self.system.bots["b2"].inbox
        assert len(inbox) == 2
        assert inbox[0]["type"] == "enemy_spotted"
        assert inbox[1]["type"] == "status"

    def test_case_12(self):
        """通信有延迟：消息不是瞬间到达，延迟结束后才投递"""
        self.system.add_bot("b1")
        self.system.add_bot("b2")
        self.system.bot_communication("b1", "hello", msg_type="status", delay=0.5)
        self.system.update(0.2)
        assert self.system.bots["b2"].inbox == []
        self.system.update(0.4)
        assert len(self.system.bots["b2"].inbox) == 1

    # ---------- 难度系统 ----------

    def test_case_13(self):
        """难度真实影响属性：简单与困难的血量/伤害/速度不同"""
        easy = self.system.difficulty_system("easy")
        hard = self.system.difficulty_system("hard")
        assert easy["health"] != hard["health"]
        assert easy["damage"] != hard["damage"]
        assert easy["speed"] != hard["speed"]
        easy_bot = self.system.add_bot("e", difficulty="easy")
        hard_bot = self.system.add_bot("h", (5.0, 0.0), difficulty="hard")
        assert easy_bot.max_health != hard_bot.max_health

    def test_case_14(self):
        """难度影响反应时间：困难bot比简单bot更快进入攻击"""
        self.system.add_bot("easy", difficulty="easy")
        self.system.add_bot("hard", (0.0, 5.0), difficulty="hard")
        self.system.add_player("p1", (10.0, 0.0))
        self.system.update(0.3)
        # 困难bot已反应，简单bot还在反应时间内
        assert self.system.bots["hard"].behavior == "attack"
        assert self.system.bots["easy"].behavior != "attack"
        # 足够时间后简单bot也能反应
        self.system.update(0.7)
        assert self.system.bots["easy"].behavior == "attack"

    def test_case_15(self):
        """难度影响瞄准精度：困难bot命中率高于简单bot"""
        self.system.add_bot("easy", difficulty="easy")
        self.system.add_bot("normal", (0.0, 5.0), difficulty="normal")
        self.system.add_bot("hard", (0.0, -5.0), difficulty="hard")
        p_easy = self.system.hit_probability("easy", distance=10.0)
        p_normal = self.system.hit_probability("normal", distance=10.0)
        p_hard = self.system.hit_probability("hard", distance=10.0)
        assert p_easy < p_normal < p_hard

    def test_case_16(self):
        """难度影响行为复杂度：简单bot不会撤退等高级战术"""
        easy = self.system.difficulty_system("easy")
        hard = self.system.difficulty_system("hard")
        assert easy["tactics"] != hard["tactics"]
        assert "retreat" not in easy["tactics"]
        assert "retreat" in hard["tactics"]
        # 简单bot残血也不会撤退，困难bot会
        self.system.add_bot("e", difficulty="easy")
        self.system.add_bot("h", (0.0, 5.0), difficulty="hard")
        self.system.add_player("p1", (10.0, 0.0))
        self.system.apply_damage("e", 45.0)   # 血量降至 25%
        self.system.apply_damage("h", 120.0)  # 血量降至 20%
        self.system.update(1.0)
        assert self.system.bots["e"].behavior != "retreat"
        assert self.system.bots["h"].behavior == "retreat"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
