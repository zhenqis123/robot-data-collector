# TODO

## Purchase List

- 木桌子 120*50*75 x 2 
https://item.taobao.com/item.htm?abbucket=18&id=959656134533&mi_id=0000BypIRmH_n-Oo0oR90wlkW2pC4Zf_jM6MmnSZaCpQw-o&ns=1&priceTId=2150424117645789319432288e28f3&skuId=5926381453374&spm=a21n57.1.hoverItem.6&utparam=%7B%22aplus_abtest%22%3A%220a9d0ebfa16dc85193b813bb0750faa0%22%7D&xxc=taobaoSearch
- 椅子 x 2 (如果有空的就不买)
- 脚架 x 2, 1.7m 单机位
https://detail.tmall.com/item.htm?ali_refid=a3_420434_1006%3A1666080105%3AH%3AkZ7zV5EQkheCK4UJ5QWXCJIgenMlnzZx%3A2f9acabf0859fa9c771007a928b5f2e9&ali_trackid=282_2f9acabf0859fa9c771007a928b5f2e9&id=680188096015&mi_id=0000dmkcZR7Cm3nxbL74wbI_aWn1teXjT1KfWRubSH_CD78&mm_sceneid=1_0_2949630009_0&priceTId=2150424117645791340382852e28f3&skuId=4880868553707&spm=a21n57.1.hoverItem.2&utparam=%7B%22aplus_abtest%22%3A%22d6205ad89221da1b8e3d0142ad2c5a05%22%7D&xxc=ad_ztc
- 再打一个头戴相机支架
- type-c 延长线 5m 长线 x 1 
- 有线type-c耳机 x 1 https://detail.tmall.com/item.htm?abbucket=18&fpChannel=101&fpChannelSig=6ebb0e8e5b141b6dfbb545d699fefb70a27b4ffa&id=651569770577&mi_id=0000TdtmuNIIMykA91zliBW1Ttuixt3Oo92dlmAlbrg4z9o&ns=1&priceTId=2147824d17645816860424938e1900&skuId=5779472036093&spm=a21n57.1.hoverItem.4&utparam=%7B%22aplus_abtest%22%3A%22297274d95455a5fc4e0550d7551bd1ee%22%7D&xxc=taobaoSearch
- USB公转type-c母 x 1 https://detail.tmall.com/item.htm?abbucket=18&id=809651945279&mi_id=0000X__ZPwQ6cB2XxjM9LCEySbaTajiMLOWckLKO43o0ATc&ns=1&priceTId=2147867e17645817747908380e0edc&skuId=5668377831346&spm=a21n57.1.hoverItem.5&utparam=%7B%22aplus_abtest%22%3A%2288b8927b99290edec19ebd7662660a90%22%7D&xxc=taobaoSearch
- 踏板 x 1 https://item.taobao.com/item.htm?abbucket=18&id=671465123110&mi_id=0000BWOYBDkLQ5Jp5PFuSQgJnFMDJW0J65ogQ91QPqAxaUQ&ns=1&priceTId=214781b417645803270971412e0f12&skuId=5011087697412&spm=a21n57.1.hoverItem.18&utparam=%7B%22aplus_abtest%22%3A%22feb37f72f34dbf6700f7aeddbcf03a6e%22%7D&xxc=taobaoSearch
- 还需要一台ubuntu电脑，一台电脑跑两个采集程序现在还没验证

## EgoDex Selected Task List

这份清单精选了egodex数据集中最具代表性的 20 个任务，涵盖了从精细指尖操作到双手协同工作的多种场景。

### 1. 可逆任务 (Reversible Tasks)
*此类任务包含操作及其逆过程，非常适合在桌面上反复进行数据采集。*

* [cite_start]**tie_and_untie_shoelace** [cite: 403]
    * *场景*：系/解鞋带。这是论文中明确提到的桌面灵巧操作代表性任务 [cite: 25]，涉及柔性绳索的复杂打结。
* [cite_start]**insert_remove_usb** [cite: 394]
    * *场景*：插入/拔出 USB。典型的电子设备桌面操作，要求高精度的孔位对齐。
* [cite_start]**charge_uncharge_device** [cite: 425]
    * *场景*：设备充电/拔线。涉及线缆管理和接口连接，是现代桌面的常见行为。
* [cite_start]**fold_unfold_paper_basic** [cite: 388]
    * *场景*：折纸/展开。处理桌面上的纸张，测试对薄片物体的折痕控制。
* [cite_start]**screw_unscrew_bottle_cap** [cite: 441]
    * *场景*：旋紧/旋松瓶盖。日常生活中的基础旋转操作。
* [cite_start]**zip_unzip_bag** [cite: 448]
    * *场景*：拉拉链。常用于桌面收纳袋或笔袋的操作，需要双手轨迹协同。
* [cite_start]**assemble_disassemble_legos** [cite: 419]
    * *场景*：乐高积木组装/拆卸。经典的桌面精细拼装任务，测试手指捏持力。
* [cite_start]**stack_unstack_cups** [cite: 401]
    * *场景*：堆叠/拆分杯子。测试物体嵌套和平衡堆叠的能力。
* [cite_start]**open_close_insert_remove_case** [cite: 405]
    * *场景*：开关盒子并存取物品。类似于眼镜盒或文具盒的开合与收纳。
* [cite_start]**assemble_disassemble_furniture_bench_chair** [cite: 385]
    * [cite_start]*场景*：桌面家具模型组装。源自 FurnitureBench 基准，涉及利用工具或卡扣进行部件组装 [cite: 151]。

### 2. 免重置任务 (Reset-free Tasks)
*无需重置环境即可在桌面上连续循环执行的任务。*

* **type_keyboard** [cite: 467]
    * *场景*：敲击键盘。最标准的桌面办公任务，测试多指独立点击能力。
* [cite_start]**write** [cite: 483]
    * *场景*：书写。涉及笔的握持和精细轨迹控制，是极具代表性的桌面精细操作。
* [cite_start]**flip_pages** [cite: 472]
    * *场景*：翻书。处理书本这一常见桌面物体，需要分离极薄页面的技巧。
* **stamp_paper** [cite: 479]
    * *场景*：盖章。涉及按压和定点放置，常见的办公桌文书处理动作。
* [cite_start]**use_rubiks_cube** [cite: 462]
    * *场景*：玩魔方。涉及物体在手中的重定向（In-hand manipulation），通常在桌面上方进行。
* [cite_start]**wipe_screen** [cite: 464]
    * *场景*：擦拭屏幕。对桌面显示器或平板设备进行表面清洁。

### 3. 需重置任务 (Reset Tasks)
*每次操作后需要人工重置桌面的任务。*

* **basic_pick_place** [cite: 489]
    * *场景*：基础抓取放置。机器人操作的入门基准，在桌面上移动物体。
* **sort_beads** [cite: 498]
    * *场景*：珠子分类。极高精度的微小物体抓取任务，典型的桌面精细作业。
* [cite_start]**use_chopsticks** [cite: 499]
    * *场景*：使用筷子。在桌面上通过工具夹取物体，极具挑战性。
* [cite_start]**pour** [cite: 497]
    * *场景*：倾倒。将液体或颗粒从一个容器倒入桌面上的另一个容器。

### 4. 验证任务
*可以从这个任务初步验证一下采集流程和subtask标注*。后面可以使用随机任务/固定任务组合，一个同学采随机任务（一大堆东西在桌子上VLM随机生成任务），一个同学采固定任务。可以先准备一下这些东西。

* **desktop_organization_setup_routine**
    * [cite_start]*场景*：桌面收纳与设备准备。这是一个长程组合任务，涵盖了从毫米级插拔对齐到大范围擦拭的完整操作流。该任务有序串联了以下 EgoDex 原生任务：整理散落文具（类似于 **declutter_desk** [cite: 488][cite_start]）、旋紧瓶盖（**screw_unscrew_bottle_cap** [cite: 441][cite_start]）、连接电源（**insert_remove_plug_socket** [cite: 436][cite_start]）以及最后清洁桌面（**clean_surface** [cite: 468]）。

    * **所需物品清单 (Required Objects)**：
        1.  **文具**：2-3 支不同粗细的笔（或马克笔）。
        2.  **收纳容器**：1 个笔筒（或空杯子）。
        3.  **旋盖物体**：1 个带螺旋盖的水瓶（瓶身与瓶盖初始分离）。
        4.  **接口设备**：1 个电源插排（或 USB Hub）以及 1 根对应的插头线缆（或 USB 线）。
        5.  **清洁工具**：1 块折叠好的抹布（或纸巾）。

    * **详细动作分解 (Micro-Action Steps)**：

        * **阶段 1：文具归位 (Stationery Retrieval & Placement)**
            * [cite_start]*对应任务参考：basic_pick_place [cite: 489][cite_start], declutter_desk [cite: 488]*
            * **Step 1.1**：伸手靠近桌面上散落的第一支笔，调整手腕角度以匹配笔杆方向。
            * **Step 1.2**：使用拇指和食指/中指（对指捏取）抓紧笔杆中段。
            * **Step 1.3**：垂直提起笔，移动手臂至笔筒上方约 5-10 厘米处。
            * **Step 1.4**：调整笔的姿态使其垂直于笔筒口，松开手指，让笔自然落入筒中（或受控插入）。
            * **Step 1.5**：重复上述动作，将剩余的笔逐一收入笔筒。

        * **阶段 2：容器封闭 (Container Sealing)**
            * [cite_start]*对应任务参考：screw_unscrew_bottle_cap [cite: 441]*
            * **Step 2.1**：非惯用手（如左手）抓住水瓶瓶身以固定位置。
            * **Step 2.2**：惯用手（如右手）抓取桌面的瓶盖。
            * **Step 2.3**：将瓶盖移动至瓶口上方，精细调整水平角度以对齐螺纹。
            * **Step 2.4**：下压瓶盖并进行顺时针旋转（旋紧动作），手指需要进行重抓（Regrasp）以完成多圈旋转，直至拧紧。

        * **阶段 3：电源连接 (Power Connection)**
            * [cite_start]*对应任务参考：insert_remove_plug_socket [cite: 436][cite_start], insert_remove_usb [cite: 394]*
            * **Step 3.1**：伸手抓取电源插头（或 USB 接头）的硬质头部。
            * **Step 3.2**：将插头移动至插排插孔前方，悬停并微调插头朝向（Roll/Pitch/Yaw）以匹配插孔形状。
            * **Step 3.3**：缓慢推进，感受插头导电脚进入插孔的物理对准（此时手部会有遮挡）。
            * **Step 3.4**：施加向前的推力，克服插孔内的弹片阻力，将插头完全推入到底。
            * **Step 3.5**：松开插头，手部撤回。

        * **阶段 4：区域清洁 (Surface Cleaning)**
            * [cite_start]*对应任务参考：clean_surface [cite: 468][cite_start], wipe_kitchen_surfaces [cite: 478]*
            * **Step 4.1**：移动手臂抓取桌角的抹布。
            * **Step 4.2**：手掌张开按压在抹布上（Power Grip 或 Palm Press），对刚刚腾出的桌面区域施加下压力。
            * **Step 4.3**：以“Z”字形或圆周运动轨迹擦拭桌面，手臂带动抹布覆盖约 $30cm \times 30cm$ 的区域。
            * **Step 4.4**：提起抹布，将其放置到桌面边缘或指定收纳区。
