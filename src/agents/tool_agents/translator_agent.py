from typing import Dict, Any, List, Optional
from src.agents.tool_agents.base_tool_agent import BaseToolAgent
from src.formats.latex.prompts import *
from src.formats.latex.utils import *
from pathlib import Path
import sys
import os
import re
import regex
import asyncio
import aiohttp
import requests
import time
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


base_dir = os.getcwd()
sys.path.append(base_dir)

class TranslatorAgent(BaseToolAgent):
    def __init__(self, 
                 config: Dict[str, Any], 
                 trans_mode: str = 0,
                 project_dir: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 errors_report: Optional[List[Dict]] = None,
                 ):
        super().__init__(agent_name="TranslatorAgent", config=config)
        self.config = config
        self.update_term = config.get("update_term", False)
        self.model = config["llm_config"].get("model", "gpt-4o")
        self.base_url = config["llm_config"].get("base_url", None)
        self.API_KEY = config["llm_config"].get("api_key", None)
        self.target_language = config.get("target_language", "ch")
        self.category = config.get("category", None)
        self.project_dir = project_dir  # Project path for parsing
        self.output_dir = output_dir  # Output directory for parsed files
        self.fail_section_nums = []
        self.fail_caption_phs = []
        self.fail_env_phs = []
        self.have_fail_parts = False
        self.errors_report = errors_report if errors_report is not None else []
        self.trans_mode = trans_mode if trans_mode is not None else 0
        # self.term_dict = config.get("term_dict", {})  # Dictionary for terminology translation
        self.term_dict = {}
        self.summary = ''
        self.prev_text = ''
        self.prev_transed_text = ''
        self.currant_content = ''

    async def execute(self, error_retry_count=0, Maxtry=3):
        """
        只修改了mode0部分,其他mode无需再修改execute及其上游方法,其已经改入异步循环
        但其他mode需要修改execute内部内容及其下游方法
        """

        self.build_term_dict()

        sections = self.read_file(Path(self.output_dir, "sections_map.json"), "json")
        captions = self.read_file(Path(self.output_dir, "captions_map.json"), "json")
        envs = self.read_file(Path(self.output_dir, "envs_map.json"), "json")

        if self.trans_mode == 0 or self.trans_mode == 2:
            self.log(f"🤖💬 Starting translating for project...⏳: {os.path.basename(self.project_dir)}.")

            async with aiohttp.ClientSession() as session:
                sem = asyncio.Semaphore(50)  # 考虑到api响应速度,大概10s左右处理一个section,每半秒启动一次调用,10左右应该不会浪费api token

                async def process_section(i, sec):
                    async with sem:
                        translated = await self.translate(sec, envs, captions, session)
                        return i, translated

                tasks = [process_section(i, sec) for i, sec in enumerate(sections)]

                for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Translating...",
                                   unit="section"):
                    i, translated_section = await future
                    sections[i] = translated_section

                    # 保存考虑修改为内存整合一次硬存读写
                    self.save_file(Path(self.output_dir, "sections_map.json"), "json", sections)
                    self.save_file(Path(self.output_dir, "captions_map.json"), "json", captions)
                    self.save_file(Path(self.output_dir, "envs_map.json"), "json", envs)

                await self._val_fail_parts(Maxtry=Maxtry,
                                     sections=sections,
                                     captions=captions,
                                     envs=envs,
                                     session=session)

                self.log(f"✅ Successfully translated sections!")

        elif self.trans_mode == 1:
            """
            暂时没有进入该mode1分支,故加注释
            """
            # error_parts = [error_part["num_or_ph"] for error_part in self.errors_report]
            # self.log(f"🤖💬 Starting retranslating for error parts:{error_parts}, the {error_retry_count+1} chance for {Maxtry} total.")
            # self._retranslate_error_parts(secs=sections,
            #                               caps=captions,
            #                               envs=envs)
            #
            # self.save_file(Path(self.output_dir, "sections_map.json"), "json", sections)
            # self.save_file(Path(self.output_dir, "captions_map.json"), "json", captions)
            # self.save_file(Path(self.output_dir, "envs_map.json"), "json", envs)
            #
            # self.fail_section_nums.clear()
            # self.fail_caption_phs.clear()
            # self.fail_env_phs.clear()
            # self.have_fail_parts = False
            #
            # self._val_fail_parts(Maxtry=Maxtry,
            #                      sections=sections,
            #                      captions=captions,
            #                      envs=envs)

            self.log(f"✅ Successfully retranslated error parts!")

        elif self.trans_mode == 3 or self.trans_mode == 4:
            """
            暂时没有进入该mode4分支,故加注释
            """
            # self.log(f"🤖💬 Starting translating for project...⏳: {os.path.basename(self.project_dir)}.")
            #
            # for i, section in tqdm(enumerate(sections), desc="Translating...", total=len(sections), unit="section"):
            #     try:
            #         self.currant_content = self._extract_text_from_tex(section["content"])
            #     except Exception as e:
            #         pass
            #     if not self.summary and self.currant_content:
            #         # self.prev_text = self._merge_with_prev_sections(sections, i)
            #         self.summary = self._request_llm_for_summary(get_summary_system_prompt, self.currant_content)
            #
            #     section = self.translate(section, envs, captions)
            #     sections[i] = section
            #     if self.summary and self.currant_content:
            #         self.summary = self._request_llm_for_refine_summary(refine_summary_system_prompt, self.currant_content, self.summary)
            #
            #
            #     self.save_file(Path(self.output_dir, "sections_map.json"), "json", sections)
            #     self.save_file(Path(self.output_dir, "captions_map.json"), "json", captions)
            #     self.save_file(Path(self.output_dir, "envs_map.json"), "json", envs)
            #
            # self._val_fail_parts(Maxtry=Maxtry,
            #                      sections=sections,
            #                      captions=captions,
            #                      envs=envs)
            
            self.log(f"✅ Successfully translated sections!")

    async def translate(self,
                        section: Dict[str, Any],
                        envs: List[Dict[str, Any]],
                        captions: List[Dict[str, Any]],
                        session: aiohttp.ClientSession) -> Dict[str, Any]:
        """
        Translates the input data (异步)
        """
        placeholder_pattern_cap = r"<PLACEHOLDER_CAP_\d+>"
        placeholder_pattern_env = r"<PLACEHOLDER_ENV_\d+>"
        placeholders_cap = re.findall(placeholder_pattern_cap, section["content"])
        placeholders_env = re.findall(placeholder_pattern_env, section["content"])

        section = await self._translate_section(section, session)  # 注意异步调用

        for placeholder in placeholders_env:
            for i, env in enumerate(envs):
                if placeholder == env["placeholder"]:
                    placeholders_cap_in_env = re.findall(placeholder_pattern_cap, env["content"])
                    placeholders_cap.extend(placeholders_cap_in_env)
                    envs[i] = await self._translate_env(env, session)  #目前象征性修改,env方法好像没有使用,其下游方法没有异步修改
                    break

        # remove duplicates
        placeholders_cap = list(dict.fromkeys(placeholders_cap))

        for placeholder in placeholders_cap:
            for i, caption in enumerate(captions):
                if placeholder == caption["placeholder"]:
                    captions[i] = await self._translate_caption(caption, session)  # 异步翻译标题
                    break

        return section
    
    async def _val_fail_parts(self, sections, captions, envs, Maxtry, session: aiohttp.ClientSession, fail_retry_count=0) -> str:
            while fail_retry_count < Maxtry and self.have_fail_parts:
                fail_parts = self.fail_section_nums + self.fail_caption_phs + self.fail_env_phs
                if fail_retry_count == Maxtry:  #  retry 3 times
                    print(f"❌ Failed to translate {fail_parts}")
                    break
                self.log(f"🤖💬 Starting retranslating for fail parts:{fail_parts}, the {fail_retry_count+1} chance for {Maxtry} total.")

                await self._retranslate_fail_parts(secs=sections,
                                            caps=captions,
                                            envs=envs,
                                            session=session)
                self.save_file(Path(self.output_dir, "sections_map.json"), "json", sections)
                self.save_file(Path(self.output_dir, "captions_map.json"), "json", captions)
                self.save_file(Path(self.output_dir, "envs_map.json"), "json", envs)
                
                fail_retry_count += 1

    async def _retranslate_fail_parts(self,
                                secs: List[Dict[str, Any]], 
                                caps: List[Dict[str, Any]], 
                                envs: List[Dict[str, Any]],
                                session: aiohttp.ClientSession) -> Any:
        sec_nums = self.fail_section_nums[:]
        cap_phs = self.fail_caption_phs[:]
        env_phs = self.fail_env_phs[:]
        self.fail_section_nums.clear()
        self.fail_caption_phs.clear()
        self.fail_env_phs.clear()
        self.have_fail_parts = False

        sec_dict = {s["section"]: i for i, s in enumerate(secs)}
        cap_dict = {c["placeholder"]: i for i, c in enumerate(caps)}
        env_dict = {e["placeholder"]: i for i, e in enumerate(envs)}

        if sec_nums:
            self.log(f"Retranslating for {sec_nums}")
            for sec_num in sec_nums:
                if sec_num == -1 or sec_num == 0:
                    continue
                if sec_num in sec_dict:
                    i = sec_dict[sec_num]
                    secs[i] = await self._translate_section(secs[i], session)
            # else:
            #     print(f"[Warning] Section {sec_num} not found.")
        if cap_phs:
            self.log(f"Retranslating for {cap_phs}")
            for cap_ph in cap_phs:
                if cap_ph in cap_dict:
                    i = cap_dict[cap_ph]
                    caps[i] = await self._translate_caption(caps[i], session) # env和caption也需要改
            # else:
            #     print(f"[Warning] Caption placeholder {cap_ph} not found.")
        if env_phs:
            self.log(f"Retranslating for {env_phs}")
            for env_ph in env_phs:
                if env_ph in env_dict:
                    i = env_dict[env_ph]
                    envs[i] = await self._translate_env(envs[i], session) # env和caption也需要改
            # else:
            #     print(f"[Warning] Environment placeholder {env_ph} not found.")

    def _retranslate_error_parts(self, secs, caps, envs) -> Any:
        for error_report in tqdm(self.errors_report, desc="ReTranslating...", unit="part"):
            error_message = []
            if "command_error" in error_report:
                error_message.append(error_report["command_error"])
            if "ph_error" in error_report:
                error_message.append(error_report["ph_error"])
            if "bracket_error" in error_report:
                error_message.append(error_report["bracket_error"])
            error_message = "\n".join(error_message)

            if error_report["part"] == "sec":
                for i, sec in enumerate(secs):
                    if error_report["num_or_ph"] == sec["section"]:
                        secs[i] = self._translate_section(section=sec, error_message=error_message)

            elif error_report["part"] == "env":
                for i, env in enumerate(envs):
                    if error_report["num_or_ph"] == env["placeholder"]:
                        envs[i] = self._translate_env(env=env, error_message=error_message)

            elif error_report["part"] == "cap":
                for i, cap in enumerate(caps):
                    if error_report["num_or_ph"] == cap["placeholder"]:
                        caps[i] = self._translate_caption(caption=cap, error_message=error_message)
            else:
                continue

    async def _translate_section(self, section: Dict[str, Any], session: aiohttp.ClientSession, error_message=None) -> \
    Dict[str, Any]:
        """只修改了mode0的异步操作,后续mode的修改需要把对应request方法也修改"""
        transed_section = section.copy()
        section_num = section["section"]
        if self.trans_mode == 0:
            # 异步请求api部分
            transed_section["trans_content"] = await self._request_llm_for_trans(
                section_system_prompt,
                section["content"],
                fail_part=section_num,
                type="sec",
                session=session
            )
        elif self.trans_mode == 1:
            transed_section["trans_content"] = self._request_llm_for_retrans_error_parts(retrans_error_parts_system_prompt,
                                                                                         part=transed_section,
                                                                                         error_message=error_message,
                                                                                         fail_part=section_num,
                                                                                         type="sec")
        elif self.trans_mode == 2:
            """
            修改mode0和2为异步并发,包括
            _request_llm_for_trans_with_terms方法 _done
            _extract_text_from_tex方法 _wait
            _request_llm_for_extract_terms方法 _wait
            """
            if not self.term_dict:
                transed_section["trans_content"] = await self._request_llm_for_trans(
                    section_system_prompt,
                    section["content"],
                    fail_part=section_num,
                    type="sec",
                    session=session
                )
            else:
                transed_section["trans_content"] = await self._request_llm_for_trans_with_terms(
                                                            section_system_prompt_with_dict,
                                                            section["content"], 
                                                            fail_part=section_num,
                                                            type="sec",
                                                            session=session
                                                            )
                
            try:
                src_text = self._extract_text_from_tex(transed_section["content"])
                tgt_text = self._extract_text_from_tex(transed_section["trans_content"])
                term_text = await self._request_llm_for_extract_terms(extract_terminology_system_prompt,
                                                        src_text,
                                                        tgt_text,
                                                        session=session
                                                        )
                if self.update_term == True:
                    # self._updated_term_dict(term_text)
                    self._updated_term_dict_v2(term_text)
            except Exception as e:
                return transed_section
            
        elif self.trans_mode == 3:
            if not self.summary:
            # if not self.prev_text or not self.prev_transed_text:
                transed_section["trans_content"] = self._request_llm_for_trans(section_system_prompt, 
                                                            section["content"], 
                                                            fail_part=section_num,
                                                            type="sec"
                                                            )
            else:
                transed_section["trans_content"] = self._request_llm_for_trans_with_sum(section_system_prompt_with_sum, 
                                                            section["content"], 
                                                            fail_part=section_num,
                                                            type="sec"
                                                            )
                
        elif self.trans_mode == 4:
            # if not self.prev_text or not self.prev_transed_text or not self.term_dict:
            if not self.summary or not self.term_dict:
                transed_section["trans_content"] = self._request_llm_for_trans(section_system_prompt, 
                                                            section["content"], 
                                                            fail_part=section_num,
                                                            type="sec"
                                                            )
            else:
                transed_section["trans_content"] = self._request_llm_for_trans_with_terms_sum(section_system_prompt_with_terms_sum, 
                                                            section["content"], 
                                                            fail_part=section_num,
                                                            type="sec"
                                                            )
            try:
                src_text = self._extract_text_from_tex(transed_section["content"])
                tgt_text = self._extract_text_from_tex(transed_section["trans_content"])
                term_text = self._request_llm_for_extract_terms(extract_terminology_system_prompt,
                                                        src_text,
                                                        tgt_text)
                if self.update_term == True:
                    # self._updated_term_dict(term_text)
                    self._updated_term_dict_v2(term_text)
            except Exception as e:
                return transed_section

        return transed_section

    async def _translate_caption(self, caption: Dict[str, Any], session: aiohttp.ClientSession, error_message=None) -> Dict[str, Any]:
        """
        Translates the captions of the input data.
        """
        transed_caption = caption.copy()
        placeholder = caption["placeholder"]
        if self.trans_mode == 0 or self.trans_mode == 3:
            transed_caption["trans_content"] = await self._request_llm_for_trans(caption_system_prompt,
                                                        caption["content"],
                                                        fail_part=placeholder,
                                                        type="cap",
                                                        session=session
                                                        )
        elif self.trans_mode == 1:
            """先不改"""
            print("translate_caption_mode_1")
            transed_caption["trans_content"] = self._request_llm_for_retrans_error_parts(retrans_error_parts_system_prompt,
                                                                                         part=transed_caption,
                                                                                         error_message=error_message,
                                                                                         fail_part=placeholder,
                                                                                         type="cap")
            
        elif self.trans_mode == 2 or self.trans_mode == 4:
            if not self.term_dict:
                transed_caption["trans_content"] = await self._request_llm_for_trans(caption_system_prompt,
                                                        caption["content"], 
                                                        fail_part=placeholder,
                                                        type="cap",
                                                        session=session
                                                        )
            else:
                transed_caption["trans_content"] = await self._request_llm_for_trans_with_terms(caption_system_prompt_with_dict,
                                                                                          caption["content"],
                                                                                          fail_part=placeholder,
                                                                                          type="cap",
                                                                                          session=session)
            try:
                src_text = self._extract_text_from_tex(transed_caption["content"])
                tgt_text = self._extract_text_from_tex(transed_caption["trans_content"])
                term_text = await self._request_llm_for_extract_terms(extract_terminology_system_prompt,
                                                        src_text,
                                                        tgt_text,
                                                        session=session
                                                        )
                if self.update_term == True:
                    # self._updated_term_dict(term_text)
                    self._updated_term_dict_v2(term_text)
            except Exception as e:
                return transed_caption

        # elif self.trans_mode == 3:
        #     if not self.summary:
        #         transed_caption["trans_content"] = self._request_llm_for_trans(caption_system_prompt, 
        #                                                 caption["content"], 
        #                                                 fail_part=placeholder,
        #                                                 type="cap"
        #                                                 )
        #     else:
        #         transed_caption["trans_content"] = self._request_llm_for_trans_with_sum(caption_system_prompt_with_sum,
        #                                                                                   caption["content"],
        #                                                                                   fail_part=placeholder,
        #                                                                                   type="cap")

        return transed_caption

    async def _translate_env(self, env: Dict[str, Any], session: aiohttp.ClientSession, error_message=None) -> Dict[str, Any]:
        """
        Translates an environment block (env) based on whether translation is needed.
        """
        transed_env = env.copy()
        placeholder = env["placeholder"]
        if self.trans_mode == 0 or self.trans_mode == 3: # sum
            if env["need_trans"]:
                transed_env["trans_content"] = await self._request_llm_for_trans(env_system_prompt,
                                                            env["content"], 
                                                            fail_part=placeholder,
                                                            type="env",
                                                            session=session
                                                            )                
            else:
                transed_env["trans_content"] = env["content"]
        elif self.trans_mode == 1:
                transed_env["trans_content"] = self._request_llm_for_retrans_error_parts(retrans_error_parts_system_prompt,
                                                                                         part=transed_env,
                                                                                         error_message=error_message,
                                                                                         fail_part=placeholder,
                                                                                         type="env")
        elif self.trans_mode == 2 or self.trans_mode == 4: # dict or sum+dict
            if not self.term_dict:
                if env["need_trans"]:
                    transed_env["trans_content"] = await self._request_llm_for_trans(env_system_prompt,
                                                            env["content"], 
                                                            fail_part=placeholder,
                                                            type="env",
                                                            session=session
                                                            )
                else:
                    transed_env["trans_content"] = env["content"]
            else:
                if env["need_trans"]:
                    transed_env["trans_content"] = await self._request_llm_for_trans_with_terms(env_system_prompt_with_dict,
                                                                                            env["content"],
                                                                                            fail_part=placeholder,
                                                                                            type="env",
                                                                                            session=session)
                else:
                    transed_env["trans_content"] = env["content"]

            if env["need_trans"]:
                try:
                    src_text = self._extract_text_from_tex(transed_env["content"])
                    tgt_text = self._extract_text_from_tex(transed_env["trans_content"])
                    text = await self._request_llm_for_extract_terms(extract_terminology_system_prompt,
                                                            src_text,
                                                            tgt_text,
                                                            session=session
                                                            )
                    if self.update_term == True:
                        # self._updated_term_dict(term_text)
                        self._updated_term_dict_v2(text)
                except Exception as e:
                    return transed_env
                
        # elif self.trans_mode == 3:
        #     if not self.summary:
        #         if env["need_trans"]:
        #             transed_env["trans_content"] = self._request_llm_for_trans(env_system_prompt, 
        #                                                     env["content"], 
        #                                                     fail_part=placeholder,
        #                                                     type="env"
        #                                                     )
        #         else:
        #             transed_env["trans_content"] = env["content"]
        #     else:
        #         if env["need_trans"]:
        #             transed_env["trans_content"] = self._request_llm_for_trans_with_sum(env_system_prompt_with_sum,
        #                                                                                     env["content"],
        #                                                                                     fail_part=placeholder,
        #                                                                                     type="env")
        #         else:
        #             transed_env["trans_content"] = env["content"]

        return transed_env

    async def _request_llm_for_trans(self,
                                     system_prompt: str,
                                     text: str,
                                     fail_part: str,
                                     type: str,
                                     session: aiohttp.ClientSession) -> str:
        """修改后的异步版本,注意其上游函数都需要异步运行,整个流程需要进入异步循环"""
        payload = {
            "model": f"{self.model}",
            "messages": [
                {"role": "system", "content": f"{system_prompt}"},
                {"role": "user", "content": f"{text}"}
            ],
            "temperature": 0.7,
            "max_new_tokens": 8192
        }

        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }

        for attempt in range(1, 4):
            try:
                async with session.post(self.base_url, json=payload, headers=headers, timeout=100) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["choices"][0]["message"]["content"].strip()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < 3:
                    await asyncio.sleep(5)
                else:
                    self.have_fail_parts = True
                    if type == 'sec':
                        self.fail_section_nums.append(fail_part)
                    elif type == 'cap':
                        self.fail_caption_phs.append(fail_part)
                    else:
                        self.fail_env_phs.append(fail_part)

                    print(f"❌ Failed to translate text, return the original text:{fail_part}. {e}")
                    return text

    async def _request_llm_for_trans_with_terms(self,
                                          system_prompt: str,
                                          text: str,
                                          fail_part: str,
                                          type: str,
                                          session: aiohttp.ClientSession) -> str:

        payload = {
            "model": f"{self.model}",
            "messages": [
                {
                    "role": "system",
                    "content": f"{system_prompt}\nWhen translating, you must strictly use the following glossary for substitution. This is the highest priority rule to ensure the consistency of terms throughout the text.\n<Glossary>:\n{self.term_dict}\nNow, please translate the following new paragraph. Maintain the terminology from the glossary provided."
                },
                {
                    "role": "user",
                    "content": f"[Current LaTeX Paragraph]:\n{text}"
                }
            ],
            "temperature": 0.7,
            # "max_length": 100000,
            "max_new_tokens": 8192
        }

        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }

        for attempt in range(1, 4):
            try:
                async with session.post(self.base_url, json=payload, headers=headers, timeout=100) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["choices"][0]["message"]["content"].strip()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < 3:
                    await asyncio.sleep(5)
                else:
                    self.have_fail_parts = True
                    if type == 'sec':
                        self.fail_section_nums.append(fail_part)
                    elif type == 'cap':
                        self.fail_caption_phs.append(fail_part)
                    else:
                        self.fail_env_phs.append(fail_part)

                    print(f"❌ Failed to translate text, return the original text:{fail_part}. {e}")
                    return text

    def _request_llm_for_retrans_error_parts(self, 
                                             system_prompt: str, 
                                             part: Dict[str, Any], 
                                             error_message: str, 
                                             fail_part: str, 
                                             type: str) -> str:
        
        user_prompt = f"[Original]:\n{part['content']}\n[Translation]:\n{part['trans_content']}\n[Error]:\n{error_message}"
        # print(user_prompt,'\n')
        payload = {
            "model": f"{self.model}",
            "messages": [
                {
                    "role": "system", 
                    "content": f"{system_prompt}"
                },
                {
                    "role": "user", 
                    "content": f"{user_prompt}"
                }
            ],
            "temperature": 0.7,
            # "max_length": 100000,
            "max_new_tokens": 8192
        }

        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }
        
        
        for attempt in range(1, 4):
            try:
                response = requests.post(self.base_url, json=payload, headers=headers, timeout=100)
                response.raise_for_status()  
                result = response.json()
                # print(result["choices"][0]["message"]["content"].strip())
                return result["choices"][0]["message"]["content"].strip()
            except requests.exceptions.RequestException as e:
                # print(f"⚠️ The {attempt}th request to translate {fail_part} failed: {e}")
                if attempt < 3:
                    time.sleep(5)  
                else:
                    self.have_fail_parts = True
                    if type == 'sec':
                        self.fail_section_nums.append(fail_part)
                    elif type == 'cap':
                        self.fail_caption_phs.append(fail_part)
                    else :
                        self.fail_env_phs.append(fail_part)

                    print(f"❌ Failed to translate text, return the original text:{fail_part}. {e}")
                    return part["trans_content"]


    def _request_llm_for_trans_with_terms_sum(self, 
                               system_prompt: str, 
                               text: str, 
                               fail_part: str, 
                               type: str):
        
        payload = {
            "model": f"{self.model}",
            "messages": [
                {
                    "role": "system", 
                    "content": f"{system_prompt}"
                },
                {
                    "role": "user", 
                    "content": f"<term dictionary>:(Use these term mappings exactly.)\n{self.term_dict}\n<summary>:(This is the evolving semantic summary to help you understand the overall context.)\n{self.summary}\n<LaTeX text>:\n{text}\n<LaTeX text trasnlation>:"
                }
            ],
            "temperature": 0.7,
            # "max_length": 100000,
            "max_new_tokens": 8192
        }

        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }

        # if fail_part == "1":
        #     print(f"<dictionary of proper nouns>\n{self.term_dict}\n<summary>\n{self.summary}\n<LaTeX text>\n{text}")
        
        for attempt in range(1, 4):
            try:
                response = requests.post(self.base_url, json=payload, headers=headers, timeout=100)
                response.raise_for_status()  
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            except requests.exceptions.RequestException as e:
                # print(f"⚠️ The {attempt}th request to translate {fail_part} failed: {e}")
                if attempt < 3:
                    time.sleep(5)  
                else:
                    self.have_fail_parts = True
                    if type == 'sec':
                        self.fail_section_nums.append(fail_part)
                    elif type == 'cap':
                        self.fail_caption_phs.append(fail_part)
                    else :
                        self.fail_env_phs.append(fail_part)

                    print(f"❌ Failed to translate text, return the original text:{fail_part}. {e}")
                    return text      

    def _request_llm_for_trans_with_sum(self, 
                               system_prompt: str, 
                               text: str, 
                               fail_part: str, 
                               type: str):
        
        payload = {
            "model": f"{self.model}",
            "messages": [
                {
                    "role": "system", 
                    "content": f"{system_prompt}"
                },
                {
                    "role": "user", 
                    "content": f"<summary>: (This is the evolving semantic summary to help you understand the overall context.)\n{self.summary}\n<LaTeX text>:\n{text}\n<LaTeX text translation>:"
                }
            ],
            "temperature": 0.7,
            # "max_length": 100000,
            "max_new_tokens": 8192
        }

        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }

        # if fail_part == "1":
        #     print(f"<summary>\n{self.summary}\n<LaTeX text>\n{text}")
        
        for attempt in range(1, 4):
            try:
                response = requests.post(self.base_url, json=payload, headers=headers, timeout=100)
                response.raise_for_status()  
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            except requests.exceptions.RequestException as e:
                # print(f"⚠️ The {attempt}th request to translate {fail_part} failed: {e}")
                if attempt < 3:
                    time.sleep(5)  
                else:
                    self.have_fail_parts = True
                    if type == 'sec':
                        self.fail_section_nums.append(fail_part)
                    elif type == 'cap':
                        self.fail_caption_phs.append(fail_part)
                    else :
                        self.fail_env_phs.append(fail_part)

                    print(f"❌ Failed to translate text, return the original text:{fail_part}. {e}")
                    return text  
                
    def _request_llm_for_trans_with_prev(self, 
                               system_prompt: str, 
                               text: str, 
                               fail_part: str, 
                               type: str):
        
        payload = {
            "model": f"{self.model}",
            "messages": [
                {
                    "role": "system", 
                    "content": f"{system_prompt}\n<previous text>:\n{self.prev_text}\n<previous translation>:\n{self.prev_transed_text}\nNow, please translate the following new paragraph. Maintain the tone, style, and terminology from the context provided."
                },
                {
                    "role": "user", 
                    "content": f"[Current LaTeX Paragraph]:\n{text}"
                }
            ],
            "temperature": 0.7,
            # "max_length": 100000,
            "max_new_tokens": 8192
        }

        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }

        # if fail_part == "1":
        #     print(f"<summary>\n{self.summary}\n<LaTeX text>\n{text}")
        
        for attempt in range(1, 4):
            try:
                response = requests.post(self.base_url, json=payload, headers=headers, timeout=100)
                response.raise_for_status()  
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            except requests.exceptions.RequestException as e:
                # print(f"⚠️ The {attempt}th request to translate {fail_part} failed: {e}")
                if attempt < 3:
                    time.sleep(5)  
                else:
                    self.have_fail_parts = True
                    if type == 'sec':
                        self.fail_section_nums.append(fail_part)
                    elif type == 'cap':
                        self.fail_caption_phs.append(fail_part)
                    else :
                        self.fail_env_phs.append(fail_part)

                    print(f"❌ Failed to translate text, return the original text:{fail_part}. {e}")
                    return text  
                
    def _request_llm_for_trans_with_prev_terms(self, 
                               system_prompt: str, 
                               text: str, 
                               fail_part: str, 
                               type: str):
        
        payload = {
            "model": f"{self.model}",
            "messages": [
                {
                    "role": "system", 
                    "content": f"{system_prompt}\nWhen translating, you must strictly use the following glossary for substitution. This is the highest priority rule to ensure the consistency of terms throughout the text.\n<Glossary>:\n{self.term_dict}\nTo ensure consistency in terminology and style, here is the context of the preceding paragraph:\n<previous text>:\n{self.prev_text}\n<previous translation>:\n{self.prev_transed_text}\nNow, please translate the following new paragraph. Maintain the tone, style, and terminology from the context provided."
                },
                {
                    "role": "user", 
                    "content": f"[Current LaTeX Paragraph]:\n{text}"
                }
            ],
            "temperature": 0.7,
            # "max_length": 100000,
            "max_new_tokens": 8192
        }

        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }

        # if fail_part == "1":
        #     print(f"<summary>\n{self.summary}\n<LaTeX text>\n{text}")
        
        for attempt in range(1, 4):
            try:
                response = requests.post(self.base_url, json=payload, headers=headers, timeout=100)
                response.raise_for_status()  
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            except requests.exceptions.RequestException as e:
                # print(f"⚠️ The {attempt}th request to translate {fail_part} failed: {e}")
                if attempt < 3:
                    time.sleep(5)  
                else:
                    self.have_fail_parts = True
                    if type == 'sec':
                        self.fail_section_nums.append(fail_part)
                    elif type == 'cap':
                        self.fail_caption_phs.append(fail_part)
                    else :
                        self.fail_env_phs.append(fail_part)

                    print(f"❌ Failed to translate text, return the original text:{fail_part}. {e}")
                    return text  
             
    async def _request_llm_for_extract_terms(self, system_prompt, src, tgt,
                                       session: aiohttp.ClientSession) -> str:

        payload = {
            "model": f"{self.model}",
            "messages": [
                {
                    "role": "system", 
                    "content": f"{system_prompt}"
                },
                {
                    "role": "user", 
                    "content": f"<en source>\n{src}\n<zh translation>\n{tgt}"
                }
            ],
            "temperature": 0.7,
            # "max_length": 100000,
            # "max_tokens": 50
        }

        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }

        for attempt in range(1, 4):
            try:
                async with session.post(self.base_url, json=payload, headers=headers, timeout=100) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["choices"][0]["message"]["content"].strip()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < 3:
                    await asyncio.sleep(5)
                else:
                    print(f"⚠️ Failed to extract terms, set N/A.")
                    return "N/A"

    def _request_llm_for_summary(self, system_prompt: str, text: str) -> str:
        """
        Requests the LLM to summarize the given text.
        """
        payload = {
            "model": f"{self.model}",
            "messages": [
                {
                    "role": "system", 
                    "content": f"{system_prompt}"
                },
                {
                    "role": "user", 
                    "content": f"<Text to summarize>:\n{text}\n<Summary>:\n"
                }
            ],
            "temperature": 0.7,
            # "max_length": 100000,
            "max_new_tokens": 8192
        }

        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }
        
        for attempt in range(1, 4):
            try:
                response = requests.post(self.base_url, json=payload, headers=headers, timeout=100)
                response.raise_for_status()  
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            except requests.exceptions.RequestException as e:
                if attempt < 3:
                    print(f"{e}")
                    time.sleep(3)  
                else:
                    print(f"⚠️ Failed to summarize text, set N/A.")
                    return "N/A"

    def _request_llm_for_refine_summary(self, system_prompt: str, text: str, sum: str) -> str:
        """
        Requests the LLM to refine the given summary.
        """
        payload = {
            "model": f"{self.model}",
            "messages": [
                {
                    "role": "system", 
                    "content": f"{system_prompt}"
                },
                {
                    "role": "user", 
                    "content": f"<prev_summary>:\n{sum}\n<new_section>:\n{text}\n<refined_summary>:\n"
                }
            ],
            "temperature": 0.7,
            # "max_length": 100000,
            "max_new_tokens": 8192
        }

        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }
        
        for attempt in range(1, 4):
            try:
                response = requests.post(self.base_url, json=payload, headers=headers, timeout=100)
                response.raise_for_status()  
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            except requests.exceptions.RequestException as e:
                if attempt < 3:
                    print(f"{e}")
                    time.sleep(3)  
                else:
                    print(f"⚠️ Failed to refine summary, set N/A.")
                    return "N/A"

    def _updated_term_dict(self, text: str) -> None:
        """
        Updates the term dictionary with new terms.
        """
        pattern = r'"([^"]+)"\s*-\s*"([^"]+)"'
        matches = re.findall(pattern, text)

        seen_lower = {k.lower() for k in self.term_dict}
        
        for en, zh in matches:
            en_lower = en.lower()
            if en_lower not in seen_lower:
                self.term_dict[en] = zh  # 保留原始拼写
                seen_lower.add(en_lower)

        self.save_file(Path(self.output_dir, "term_dict.json"), "json", self.term_dict)

    def _updated_term_dict_v2(self, text: str) -> None:

        new_term_dict = {}
        lines = text.split('\n')[1:]
        for line in lines:
            line = line.strip()
            if not line:
                continue  # 跳过空行

            # 使用正则匹配 "英文" - "中文" 格式
            match = re.match(r'^"(.+?)"\s*-\s*"(.+?)"$', line)
            if match:
                english = match.group(1)
                chinese = match.group(2)
                new_term_dict[english] = chinese

        for en, zh in new_term_dict:
            if en not in self.term_dict:
                self.term_dict[en] = zh


    def _process_latex_to_eva(self, latex_code):
        latex_code = replace_href(latex_code)
        latex_code = replace_includegraphics(latex_code)
        return latex_code

    def _extract_text_from_tex(self, tex):
        # convert = CustomLatexNodes2Text()
        # text = convert.latex_to_text(tex)
        tex = self._process_latex_to_eva(tex)
        text = LatexNodes2Text().latex_to_text(tex)
        text = delete_ph(text)
        return text
    
    def _merge_with_prev_sections(self, sections: list[dict], idx: int) -> str:
        """
        Merge content of current section with previous two sections (if valid).
        Ignore sections whose 'section' field is "-1" or "0".

        Parameters:
            sections (list of dict): A list of sections, each with keys "section" and "content".
            idx (int): The index of the current section in the list.

        Returns:
            str: The merged content string.
        """
        if not (0 <= idx < len(sections)):
            raise IndexError("Index out of range.")

        merged_content = []
        merged_trans_content = []

        # Check second previous section
        # if idx >= 2:
        #     sec = sections[idx - 2]
        #     if sec["section"] not in {"-1", "0"}:
        #         try:
        #             content = self._extract_text_from_tex(sec["content"])
        #             transed_content = self._extract_text_from_tex(sec["trans_content"])
        #             merged_content.append(content)
        #             merged_trans_content.append(transed_content)
        #         except Exception as e:
        #             pass
                

        # Check first previous section
        if idx >= 1:
            sec = sections[idx - 1]
            if sec["section"] not in {"-1", "0"}:
                try:
                    content = self._extract_text_from_tex(sec["content"])
                    transed_content = self._extract_text_from_tex(sec["trans_content"])
                    merged_content.append(content)
                    merged_trans_content.append(transed_content)
                except Exception as e:
                    pass

        # Always include current section
        try:
            content = self._extract_text_from_tex(sections[idx]["content"])
            transed_content = self._extract_text_from_tex(sections[idx]["trans_content"])
            merged_content.append(content)
            merged_trans_content.append(transed_content)
        except Exception as e:
            pass

        return "\n".join(merged_content)


    def build_term_dict(self):
        arxiv_id = os.path.basename(self.project_dir)
        if self.category.get(arxiv_id):
            term_dict_loaded = False
            for category in self.category[arxiv_id]:
                file_path = os.path.join('terms', f'{category}.csv')
                try:
                    df = pd.read_csv(file_path, header=None, names=['English Term', 'Chinese Translation'])
                    self.term_dict = dict(zip(df['English Term'], df['Chinese Translation']))
                    term_dict_loaded = True

                except FileNotFoundError:
                    continue

            if not term_dict_loaded:
                try:
                    df = pd.read_csv('terms/default.csv', header=None,
                                     names=['English Term', 'Chinese Translation'])
                    self.term_dict = dict(zip(df['English Term'], df['Chinese Translation']))
                except FileNotFoundError as e:
                    print(f"Error: Default terminology file not found: {e}")
        else:
            try:
                df = pd.read_csv('terms/default.csv', header=None,
                                 names=['English Term', 'Chinese Translation'])
                self.term_dict = dict(zip(df['English Term'], df['Chinese Translation']))
            except FileNotFoundError as e:
                print(f"Error: Default terminology file not found: {e}")
